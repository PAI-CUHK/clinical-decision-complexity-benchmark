#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified pipeline for evaluating medical NP problems with model solvers.

Each problem has up to ten scale levels."""

import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import MedicalNPBenchmark
from core.tasks import TASKS, legacy_task_name, resolve_task
from tasks.pra_cc.adapter import PASPAdapter
from tasks.rr_iapd.adapter import ECGPatternAdapter
from tasks.scs.adapter_o import ORSchedulingAdapter
from tasks.scs.adapter_d import ORSchedulingNPCompleteAdapter
from tasks.ed_tqm.adapter import EDSchedulingAdapter
from tasks.hnr_d.adapter import NurseSchedulingNPCompleteAdapter
from tasks.ecd_rs.adapter import MedicalRiskAdapter

# Import instance generators
from tasks.pra_cc.generator import generate_pasp_instance
from tasks.rr_iapd.generator import generate_ecg_instance
from tasks.scs.generator import generate_or_instance
from tasks.ed_tqm.generator import generate_ed_instance
from tasks.hnr_d.generator import generate_nurse_scheduling_instance
from tasks.ecd_rs.generator import generate_medical_risk_instance

# Import renderers
from tasks.pra_cc.renderer import PASPLLMRenderer
from tasks.rr_iapd.renderer import ECGPatternLLMRenderer
from tasks.pc_mde.renderer_o import DiseaseDiagnosisLLMRenderer
from tasks.pc_mde.renderer_d import DiseaseDiagnosisNPCompleteRenderer
from tasks.scs.renderer_o import ORSchedulingLLMRenderer
from tasks.scs.renderer_d import ORSchedulingNPCompleteRenderer
from tasks.ed_tqm.renderer import EDSchedulingLLMRenderer
from tasks.hnr_d.renderer import NurseSchedulingLLMRenderer
from tasks.ecd_rs.renderer import MedicalRiskRenderer

# Validators are used indirectly through problem.evaluate_solution

# Import valid-rate calculation functions
from fuction import (
    get_each_problem_valid_rate,
    calculate_valid_rate_by_problem_name,
    parse_report_file,
    calculate_valid_rate_by_type,
    calculate_valid_rate_by_complexity,
    calculate_valid_rate_by_level,
    calculate_valid_rate_by_type_and_level,
    get_overall_valid_rate,
    generate_valid_rate_report,
    calculate_valid_rates
)

# Model runner
LLM_Models = None


def _pc_mde_o_adapter():
    from tasks.pc_mde.adapter_o import DiseaseDiagnosisAdapter
    return DiseaseDiagnosisAdapter()


def _pc_mde_d_adapter():
    from tasks.pc_mde.adapter_d import DiseaseDiagnosisNPCompleteAdapter
    return DiseaseDiagnosisNPCompleteAdapter()


def _generate_pc_mde_o(config):
    from tasks.pc_mde.generator_o import generate_disease_instance
    return generate_disease_instance(config)


def _generate_pc_mde_d(config):
    from tasks.pc_mde.generator_d import generate_disease_npcomplete_instance
    return generate_disease_npcomplete_instance(config)


def get_renderer(problem_name):
    """Return the renderer for the specified problem name."""
    problem_name = legacy_task_name(problem_name)
    if problem_name == "PASP":
        return PASPLLMRenderer()
    elif problem_name == "ECG Pattern Detection":
        return ECGPatternLLMRenderer()
    elif problem_name == "Disease Diagnosis":
        return DiseaseDiagnosisLLMRenderer()
    elif problem_name == "Disease Diagnosis NP-complete":
        return DiseaseDiagnosisNPCompleteRenderer()
    elif problem_name == "OR Scheduling":
        return ORSchedulingLLMRenderer()
    elif problem_name == "OR Scheduling NP-complete":
        return ORSchedulingNPCompleteRenderer()
    elif problem_name == "ED Scheduling":
        return EDSchedulingLLMRenderer()
    elif problem_name == "Nurse Scheduling":
        return NurseSchedulingLLMRenderer()
    elif problem_name == "Medical Risk":
        return MedicalRiskRenderer()
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")





ADAPTER_MAP = {
    "PASP": (PASPAdapter, generate_pasp_instance),
    "ECG Pattern Detection": (ECGPatternAdapter, generate_ecg_instance),
    "Disease Diagnosis": (_pc_mde_o_adapter, _generate_pc_mde_o),
    "Disease Diagnosis NP-complete": (_pc_mde_d_adapter, _generate_pc_mde_d),
    "OR Scheduling": (ORSchedulingAdapter, generate_or_instance),
    "OR Scheduling NP-complete": (ORSchedulingNPCompleteAdapter, generate_or_instance),
    "ED Scheduling": (EDSchedulingAdapter, generate_ed_instance),
    "Nurse Scheduling": (NurseSchedulingNPCompleteAdapter, generate_nurse_scheduling_instance),
    "Medical Risk": (MedicalRiskAdapter, generate_medical_risk_instance)
}


def generate_problem_instances(problem_name, num_levels=10, num_instances_per_level=5, output_base_dir=None):
    """Generate problem instances across scale levels.
    
    Args:
        problem_name: Task name.
        num_levels: Number of levels.
        num_instances_per_level: Number of instances per level.
        output_base_dir: Output directory.
    
    Returns:
        A list of MedicalNPProblem instances."""
    problems = []
    
    # Get the adapter and instance generator
    problem_name = legacy_task_name(problem_name)
    if problem_name not in ADAPTER_MAP:
        raise ValueError(f"Unknown problem name: {problem_name}")
    
    adapter_class, instance_generator = ADAPTER_MAP[problem_name]
    
    # Create the output directory
    if output_base_dir:
        import json
        from pathlib import Path
        
        problem_dir = Path(output_base_dir) / problem_name
        instances_dir = problem_dir / "instances"
        baselines_dir = problem_dir / "baselines"
        prompts_dir = problem_dir / "prompts"
        
        instances_dir.mkdir(parents=True, exist_ok=True)
        baselines_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)
    
    for level in range(num_levels):
        print(f"生成 {problem_name} Level {level+1}/{num_levels} 实例...")
        
        for inst_idx in tqdm(range(num_instances_per_level), desc=f"Level {level+1}"):
            # Calculate the seed
            seed = 42 + level * 100 + inst_idx
            
            # Create the adapter and generate its instance configuration
            adapter = adapter_class()
            config = adapter.generate_instance_config(level + 1)  # Levels start at 1
            config["seed"] = seed
            
            # Generate the instance
            instance = instance_generator(config)
            
            # Build the problem once
            problem = adapter.build(instance)
            instance_id = f"level_{level+1}_instance_{inst_idx}"

            if problem.instance_data.get("invalid"):
                reason = problem.instance_data.get("reason", "unknown")
                print(f"  警告: {instance_id} build 失败，跳过保存: {reason}")
                continue

            problem.name = f"{problem_name} Level {level+1}"
            problem.metadata["level"] = level + 1
            problem.metadata["seed"] = seed
            
            # Set the problem validator
            try:
                from core.validator_factory import get_validator
                validator = get_validator(problem_name)
                problem.validator = validator
            except ValueError as e:
                print(f"  警告: {str(e)}")
            
            # Generate the prompt once
            renderer = get_renderer(problem_name)
            prompt = renderer.render(problem)
            
            # Cache the prompt on the problem
            problem.prompt = prompt
            
            # Compute the baseline once
            baseline = problem.validator.calculate_baseline(problem)
            
            # Cache the baseline on the problem
            problem.baseline = baseline
            
            # Save the instance and baseline
            if output_base_dir:
                # Save the instance
                instance_data = {
                    "instance_id": instance_id,
                    "level": level + 1,
                    "seed": seed,
                    "config": config,
                    "instance": instance.__dict__ if hasattr(instance, "__dict__") else instance
                }
                instance_path = instances_dir / f"{instance_id}.json"
                with open(instance_path, "w", encoding="utf-8") as f:
                    json.dump(instance_data, f, indent=2, ensure_ascii=False)
                
                # Save the baseline
                baseline_data = {
                    "instance_id": instance_id,
                    "level": level + 1,
                    "baseline": baseline
                }
                baseline_path = baselines_dir / f"{instance_id}.json"
                with open(baseline_path, "w", encoding="utf-8") as f:
                    json.dump(baseline_data, f, indent=2, ensure_ascii=False)
                
                # Save the prompt
                prompt_data = {
                    "instance_id": instance_id,
                    "level": level + 1,
                    "prompt": prompt
                }
                prompt_path = prompts_dir / f"{instance_id}.json"
                with open(prompt_path, "w", encoding="utf-8") as f:
                    json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            
            # Save the problem object directly
            problems.append(problem)
    
    return problems


def llm_solver(problem, backend="vllm", vllm_model=None, vllm_port=8000, 
               gpt_model="gpt-5", temperature=0.0, max_tokens=16384):
    """Solve a problem using a model backend.
    
    Args:
        problem: MedicalNPProblem instance.
        backend: vllm or gpt.
        vllm_model: Gateway model name.
        vllm_port: Legacy parameter; RESPONSES_BASE_URL sets the address.
        gpt_model: GPT model name.
        temperature: Sampling temperature.
        max_tokens: Maximum generated tokens.
    
    Returns:
        A dictionary containing raw output, solution, and usage."""
    global LLM_Models
    
    # Initialize the model runner
    if LLM_Models is None:
        from fuction import LLM_Models
        LLM_Models = LLM_Models()
    
    # Get the problem name
    problem_name = problem.name.split(" Level ")[0]
    
    # Use the cached prompt to avoid regenerating it
    prompt = getattr(problem, 'prompt', None)
    
    # Generate a prompt only when it is not cached
    if prompt is None:
        renderer = get_renderer(problem_name)
        prompt = renderer.render(problem)
    
    # Build the messages
    messages = [{"role": "user", "content": prompt}]
    
    print(f"使用 LLM 求解 {problem.name}...")
    
    # Initialize usage for all exception paths
    usage = {}
    
    try:
        # Call the model
        llm_result = LLM_Models.run_model(
            messages=messages,
            backend=backend,
            temperature=temperature,
            max_tokens=max_tokens,
            vllm_port=vllm_port,
            vllm_model=vllm_model,
            gpt_model=gpt_model
        )

        # Handle dictionary returns from run_gpt/run_vllm and legacy strings
        if isinstance(llm_result, dict) and "content" in llm_result:
            raw_output = llm_result["content"]
            usage = llm_result.get("usage", {})
        else:
            raw_output = llm_result if isinstance(llm_result, str) else str(llm_result)
            usage = {}
        
        # Parse output using the validator extract_answer method
        solution = problem.validator.extract_answer(raw_output)
        
    except Exception as e:
        print(f"Error solving {problem.name}: {e}")
        raw_output = f"Error: {str(e)}"
        solution = raw_output
        # Preserve usage data if the API call succeeded
        # Keep usage empty if the API call raised an exception
    
    # Return raw output, solution, and usage
    return {
        "raw_output": raw_output,
        "solution": solution,
        "usage": usage
    }





def get_model_name(args):
    """Return the model name used to select a results directory."""
    if args.backend == "vllm":
        # Extract the model name from the path
        model_path = args.vllm_model
        # Remove path special characters to extract the model name
        model_name = os.path.basename(model_path)
        # Clean the model name, including version numbers
        model_name = model_name.replace("-", "_").replace(".", "_")
        return f"vllm_{model_name}"
    else:  # gpt
        return f"gpt_{args.gpt_model}"


def load_problem_instances(problem_name, num_levels=10, num_instances_per_level=5, output_base_dir=None):
    """Load existing problem instances without generating new ones.
    
    Args:
        problem_name: Task name.
        num_levels: Number of levels.
        num_instances_per_level: Number of instances per level.
        output_base_dir: Directory containing saved instances.
    
    Returns:
        A list of MedicalNPProblem instances."""
    problems = []
    
    # Get the adapter and instance generator
    problem_name = legacy_task_name(problem_name)
    if problem_name not in ADAPTER_MAP:
        raise ValueError(f"Unknown problem name: {problem_name}")
    
    adapter_class, _ = ADAPTER_MAP[problem_name]
    
    # Get the instance directory, supporting two layouts
    output_base = Path(output_base_dir)
    
    # Try the current layout: output_base / problem_name / instances
    problem_dir = output_base / problem_name
    instances_dir = problem_dir / "instances"
    baselines_dir = problem_dir / "baselines"
    prompts_dir = problem_dir / "prompts"
    
    # If absent, try the legacy layout: output_base / "instances" / problem_name
    if not instances_dir.exists():
        problem_dir = output_base / "instances" / problem_name
        instances_dir = problem_dir / "instances"
        baselines_dir = problem_dir / "baselines"
        prompts_dir = problem_dir / "prompts"
    
    if not instances_dir.exists() or not baselines_dir.exists() or not prompts_dir.exists():
        raise ValueError(f"实例目录不存在，请先运行不带 --skip-generate 的命令生成实例: {instances_dir}")
    
    for level in range(num_levels):
        print(f"加载 {problem_name} Level {level+1}/{num_levels} 实例...")
        
        for inst_idx in tqdm(range(num_instances_per_level), desc=f"Level {level+1}"):
            instance_id = f"level_{level+1}_instance_{inst_idx}"
            
            # Load instance data
            instance_path = instances_dir / f"{instance_id}.json"
            baseline_path = baselines_dir / f"{instance_id}.json"
            prompt_path = prompts_dir / f"{instance_id}.json"
            
            if not instance_path.exists() or not baseline_path.exists() or not prompt_path.exists():
                print(f"  警告: 实例文件 {instance_id} 不完整，跳过")
                continue
            
            # Read instance data
            with open(instance_path, "r", encoding="utf-8") as f:
                instance_data = json.load(f)
            
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
            
            # Get the seed
            seed = instance_data.get("seed", 42 + level * 100 + inst_idx)
            
            # Reuse precomputed baseline values to avoid rerunning the solver
            baseline_info = baseline_data.get("baseline", {})
            if isinstance(baseline_info, dict):
                # Use the saved values directly
                opt_value = baseline_info.get("opt_value", 0)
                k = baseline_info.get("k", 0)
                expected_answer = baseline_info.get("baseline_answer", "No")
                satisfies_hard_constraints = baseline_info.get("satisfies_hard_constraints", False)
            else:
                # Legacy format containing only a string
                opt_value = 0
                k = 0
                expected_answer = baseline_info if isinstance(baseline_info, str) else "No"
                satisfies_hard_constraints = True
            
            # Extract k from the prompt if the baseline lacks it
            if k == 0:
                prompt_text = prompt_data.get("prompt", "")
                import re
                match = re.search(r"penalty <= (\d+)", prompt_text)
                if match:
                    k = int(match.group(1))
                    print(f"  从 prompt 中提取 k 值: {k}")
            
            # Create the adapter and build the problem
            adapter = adapter_class()
            
            # Restore the instance object from its data
            instance_dict = instance_data.get("instance", {})
            
            # Create a simple instance object
            class SimpleInstance:
                pass
            
            instance = SimpleInstance()
            instance.__dict__.update(instance_dict)
            
            # Set a precomputed k on the instance to avoid recalculation
            if k > 0:
                instance.k = k
                instance.opt_value = opt_value
                instance.expected_answer = expected_answer
                instance.satisfies_hard_constraints = satisfies_hard_constraints
            
            # Build the problem
            problem = adapter.build(instance)
            problem.name = f"{problem_name} Level {level+1}"
            problem.metadata["level"] = level + 1
            problem.metadata["seed"] = seed
            
            # Set the problem validator
            try:
                from core.validator_factory import get_validator
                validator = get_validator(problem_name)
                problem.validator = validator
            except ValueError as e:
                print(f"  警告: {str(e)}")
            
            # Set the cached prompt and baseline
            problem.prompt = prompt_data.get("prompt", "")
            problem.baseline = baseline_info
            
            problems.append(problem)
    
    return problems


def main():
    """Run the benchmark command-line entry point."""
    parser = argparse.ArgumentParser(description="医学 NP 问题统一测试流程")
    
    # Instance generation arguments
    parser.add_argument("--num_levels", type=int, default=10, 
                       help="每个问题的难度级别数量")
    parser.add_argument("--num_instances", type=int, default=10, 
                       help="每个级别生成多少个实例")
    
    # Model arguments
    parser.add_argument("--backend", type=str, default="vllm",
                       choices=["vllm", "gpt"])
    parser.add_argument("--vllm_model", type=str, 
                       default="/path/to/local/model")
    parser.add_argument("--vllm_port", type=int, default=8004, help="旧版兼容参数；连接地址由 RESPONSES_BASE_URL 配置")
    parser.add_argument("--gpt_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=65536)
    
    # Other arguments
    parser.add_argument("--output_dir", type=str, default="result/all_medical_np_problems",
                       help="输出目录")
    parser.add_argument("--problems", type=str, nargs="+", default=None,
                       help="指定要运行的问题名称（可选）。如果不指定，默认运行所有问题。"
                       "可用缩写：" + ", ".join(task.abbreviation for task in TASKS) +
                       "；包含空格的名称须用引号括起来。")
    parser.add_argument("--skip-generate", action="store_true", default=False,
                       help="跳过问题实例生成，直接从已有的实例文件加载。适用于只换模型测试的场景。")
    parser.add_argument("--generate-only", action="store_true", default=False,
                       help="只生成实例，不调用 LLM 求解。适用于只生成测试数据的场景。")
    
    args = parser.parse_args()
    if args.problems:
        try:
            args.problems = [legacy_task_name(name) for name in args.problems]
        except ValueError as exc:
            parser.error(str(exc))
    
    # Get the model name for the results directory
    model_name = get_model_name(args)
    
    # Create the directory structure:
    # - instances/: instance files shared by all models
    # - results/{model_name}/: model-specific results
    output_base_dir = Path(args.output_dir)
    instances_base_dir = output_base_dir / "instances"  # Shared instance directory
    results_base_dir = output_base_dir / "results" / model_name  # Model-specific results directory
    
    instances_base_dir.mkdir(parents=True, exist_ok=True)
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== 使用模型: {model_name} ===")
    print(f"=== 实例文件位置: {instances_base_dir} ===")
    print(f"=== 结果保存位置: {results_base_dir} ===")
    
    if args.skip_generate:
        print("=== 跳过问题实例生成，从已有文件加载 ===")
    else:
        print("=== 医学 NP 问题统一测试 ===")
    
    # Define all available tasks
    all_problems = [
        "PASP",
        "ECG Pattern Detection",
        "Disease Diagnosis",
        "Disease Diagnosis NP-complete",
        "OR Scheduling",
        "OR Scheduling NP-complete",
        "ED Scheduling",
        "Nurse Scheduling",
        "Medical Risk"
    ]
    
    # Select tasks according to the arguments
    if args.problems:
        # Filter the requested tasks
        problems_to_test = [p for p in all_problems if p in args.problems]
        # Check for invalid task names
        invalid_problems = [p for p in args.problems if p not in all_problems]
        if invalid_problems:
            print(f"警告：以下问题名称无效，将被忽略: {invalid_problems}")
            print(f"有效的问题名称: {all_problems}")
        print(f"将只测试以下问题: {[resolve_task(p).abbreviation for p in problems_to_test]}")
    else:
        # Run all tasks by default
        problems_to_test = all_problems
        print(f"将测试所有 {len(problems_to_test)} 个问题")
    
    # Generate or load problems from the shared instance directory
    problems = []
    for problem_name in problems_to_test:
        if args.skip_generate:
            # Load instances from existing files
            problem_list = load_problem_instances(
                problem_name,
                num_levels=args.num_levels,
                num_instances_per_level=args.num_instances,
                output_base_dir=instances_base_dir
            )
        else:
            # Generate new instances
            problem_list = generate_problem_instances(
                problem_name,
                num_levels=args.num_levels,
                num_instances_per_level=args.num_instances,
                output_base_dir=instances_base_dir
            )
        problems.extend(problem_list)
    
    # Skip model solving in generate-only mode
    if args.generate_only:
        print("\n=== 实例生成完成，不调用 LLM 求解 ===")
        print(f"共生成 {len(problems)} 个问题实例")
        print(f"实例文件位置: {instances_base_dir}")
        return
    
    # Create the model-specific detailed results directory
    results_dir = results_base_dir / "detailed_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the benchmark with its output directory
    benchmark = MedicalNPBenchmark(problems, results_dir=str(results_dir))
    
    # Define the configured model solver
    def solver_with_params(problem):
        return llm_solver(
            problem,
            backend=args.backend,
            vllm_model=args.vllm_model,
            vllm_port=args.vllm_port,
            gpt_model=args.gpt_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    
    # Run the benchmark and save results during execution
    print("\n=== 运行基准测试 ===")
    results = benchmark.run_benchmark(solver_with_params)
    
    # Generate the report
    print("\n=== 生成测试报告 ===")
    report = benchmark.generate_report("reports/all_medical_np_problems_report.md")
    
    # Generate the complete valid-rate report using the imported functions
    print("\n=== 使用有效率计算函数生成评估报告 ===")
    report_path = "reports/all_medical_np_problems_report.md"
    
    print("1. 调用 get_each_problem_valid_rate...")
    each_problem_valid_rate = get_each_problem_valid_rate(report_path)
    
    print("2. 调用 calculate_valid_rate_by_problem_name...")
    valid_rate_by_name = calculate_valid_rate_by_problem_name(report_path)
    
    print("3. 调用 parse_report_file...")
    parsed_problems = parse_report_file(report_path)
    
    print("4. 调用 calculate_valid_rate_by_type...")
    valid_rate_by_type = calculate_valid_rate_by_type(parsed_problems)
    
    print("5. 调用 calculate_valid_rate_by_complexity...")
    valid_rate_by_complexity = calculate_valid_rate_by_complexity(parsed_problems)
    
    print("6. 调用 calculate_valid_rate_by_level...")
    valid_rate_by_level = calculate_valid_rate_by_level(parsed_problems)
    
    print("7. 调用 calculate_valid_rate_by_type_and_level...")
    valid_rate_by_type_level = calculate_valid_rate_by_type_and_level(parsed_problems)
    
    print("8. 调用 get_overall_valid_rate...")
    overall_valid_rate = get_overall_valid_rate(parsed_problems)
    
    print("9. 调用 calculate_valid_rates...")
    all_valid_rates = calculate_valid_rates(report_path)
    
    print("10. 调用 generate_valid_rate_report...")
    valid_rate_report = generate_valid_rate_report(report_path)
    
    # Save the valid-rate report
    valid_rate_report_path = "reports/valid_rate_report.md"
    with open(valid_rate_report_path, "w", encoding="utf-8") as f:
        f.write(valid_rate_report)
    print(f"\n有效率报告已保存到 {valid_rate_report_path}")
    
    # Print the evaluation summary
    print("\n=== 评估结果摘要 ===")
    summary = results["summary"]
    print(f"总问题数: {summary['total_problems']}")
    print(f"有效解决方案数: {summary['valid_solutions']} ({summary['valid_rate']:.2%})")
    print(f"平均求解时间: {summary['avg_solve_time']:.4f}s")
    print(f"平均质量分数: {summary['avg_quality']:.4f}")
    
    # Print results by problem type
    print("\n按问题类型统计:")
    for problem_type, stats in summary['by_problem_type'].items():
        print(f"  {problem_type}:")
        print(f"    数量: {stats['count']}")
        print(f"    有效率: {stats['valid_rate']:.2%}")
        print(f"    平均时间: {stats['avg_time']:.4f}s")
        print(f"    平均质量: {stats['avg_quality']:.4f}")
    
    # Print results by complexity
    print("\n按复杂度统计:")
    for complexity, stats in summary['by_complexity'].items():
        print(f"  {complexity}:")
        print(f"    数量: {stats['count']}")
        print(f"    有效率: {stats['valid_rate']:.2%}")
        print(f"    平均时间: {stats['avg_time']:.4f}s")
        print(f"    平均质量: {stats['avg_quality']:.4f}")
    
    # Print valid-rate statistics
    print("\n=== 有效率计算结果 ===")
    print(f"总体有效率: {overall_valid_rate['valid']}/{overall_valid_rate['total']} ({overall_valid_rate['valid_rate']}%)")
    
    print("\n按问题名称统计:")
    for name, stats in valid_rate_by_name.items():
        print(f"  {name}: {stats['valid']}/{stats['total']} ({stats['valid_rate']}%)")
    
    print("\n=== 测试完成 ===")
    print(f"基准测试报告已保存到 reports/all_medical_np_problems_report.md")
    print(f"有效率报告已保存到 {valid_rate_report_path}")
    print(f"详细结果已保存到 {results_dir}")
    print("\n评估结果也可通过以下方式获取:")
    print("1. 查看 reports/all_medical_np_problems_report.md 文件获取基准测试报告")
    print("2. 查看 reports/valid_rate_report.md 文件获取有效率统计报告")
    print("3. 查看 result/all_medical_np_problems/detailed_results/ 目录获取详细结果")
    print("4. 直接使用 results 变量访问评估结果数据")


if __name__ == "__main__":
    main()
