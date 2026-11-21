"""Benchmark framework for medical NP problems."""

from typing import List, Dict, Callable
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .problem import MedicalNPProblem
from .validator import MissingBaselineError


class EmptyLLMResponseError(Exception):
    """Raised to stop a benchmark run when model raw_output is empty."""
    pass


class MedicalNPBenchmark:
    """Manage and run medical NP benchmark problems and evaluate solvers."""
    
    def __init__(self, problems: List[MedicalNPProblem] = None, results_dir: str = None):
        """Initialize the benchmark with problems and an output directory."""
        self.problems = problems or []
        self.results = []
        self.results_dir = Path(results_dir) if results_dir else None
    
    def add_problem(self, problem: MedicalNPProblem):
        """Add a MedicalNPProblem to the benchmark."""
        self.problems.append(problem)
    
    def run_benchmark(self, solver: Callable, metrics: List[str] = None, parallel: bool = False, max_workers: int = 4) -> Dict:
        """Run the benchmark and save results during execution.
        
        Args:
            solver: Function taking a MedicalNPProblem and returning a solution.
            metrics: Evaluation metrics.
            parallel: Whether to run in parallel.
            max_workers: Number of worker threads.
        
        Returns:
            Benchmark results as a dictionary."""
        results = []
        
        # Ensure the results directory exists
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if parallel:
            # Parallel execution mode
            print(f"\n=== 并行模式: 使用 {max_workers} 个线程 ===")
            results = self._run_parallel(solver, max_workers)
        else:
            # Serial execution mode
            results = self._run_serial(solver)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results
        self.results = results
        
        return {
            "results": results,
            "summary": summary
        }
    
    def _run_serial(self, solver: Callable) -> List[Dict]:
        """Run the solver serially and return problem results."""
        results = []
        
        for i, problem in enumerate(self.problems):
            original_index = problem.metadata.get("original_index", i)
            print(f"\n=== Running problem {i+1}/{len(self.problems)}: {problem.name} ===")
            
            # Record solve time
            start_time = time.time()
            solver_output = solver(problem)
            end_time = time.time()
            solve_time = end_time - start_time
            
            # Stop execution if baseline generation fails
            try:
                result = self._process_result(original_index, problem, solver_output, solve_time)
                results.append(result)
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"ERROR: Failed to process result for {problem.name}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"{'='*60}")
                
                error_result = {
                    "index": original_index,
                    "problem_name": problem.name,
                    "problem_type": problem.problem_type,
                    "complexity": problem.metadata.get("complexity", "unknown"),
                    "solve_time": solve_time,
                    "valid": False,
                    "violations": ["error"],
                    "quality": 0,
                    "metrics": {},
                    "usage": {},
                    "validation_details": {},
                    "llm_raw_output": f"Error: {str(e)}",
                    "llm_answer": None,
                    "baseline": getattr(problem, 'baseline', None),
                    "baseline_comparison": {},
                    "error": str(e)
                }
                results.append(error_result)
                
                if self.results_dir:
                    try:
                        problem_name = problem.name
                        if "Diagnosis" in problem_name:
                            domain_dir = self.results_dir / "diagnosis"
                        elif "Pattern" in problem_name:
                            domain_dir = self.results_dir / "pattern_detection"
                        elif "Risk" in problem_name:
                            domain_dir = self.results_dir / "risk_assessment"
                        elif "Scheduling" in problem_name or "PASP" in problem_name:
                            domain_dir = self.results_dir / "scheduling"
                        else:
                            domain_dir = self.results_dir
                        
                        domain_dir.mkdir(parents=True, exist_ok=True)
                        result_filename = f"{original_index}_{problem_name.replace(' ', '_')}.json"
                        result_path = domain_dir / result_filename
                        with open(result_path, "w", encoding="utf-8") as f:
                            json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)
                    except Exception as save_error:
                        print(f"  警告: 保存错误结果失败: {save_error}")
        
        return results
    
    def _run_parallel(self, solver: Callable, max_workers: int) -> List[Dict]:
        """Run the solver in parallel using the given number of workers."""
        results = []
        futures = {}
        
        print(f"提交 {len(self.problems)} 个问题到线程池...")
        
        # Use a thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks that each solve, validate, and save a result
            for i, problem in enumerate(self.problems):
                original_index = problem.metadata.get("original_index", i)
                future = executor.submit(self._solve_validate_save, original_index, problem, solver)
                futures[future] = (original_index, problem)
            
            # Collect results already validated and saved by workers
            for future in tqdm(as_completed(futures), total=len(futures), desc="Solving problems"):
                try:
                    result = future.result()
                    results.append(result)
                except EmptyLLMResponseError:
                    # Cancel remaining tasks and stop when the model returns an empty response
                    print(f"\n{'='*60}")
                    print(f"FATAL: 检测到空 LLM 响应，取消剩余任务并停止运行")
                    print(f"{'='*60}")
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    print(f"\n{'='*60}")
                    print(f"ERROR: 收集结果时发生异常")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    print(f"{'='*60}")
        
        # Sort results by problem order
        results.sort(key=lambda x: x.get("index", 0))
        
        return results
    
    def _solve_with_timer(self, index: int, problem: MedicalNPProblem, solver: Callable) -> tuple:
        """Time one solver call and return (solver_output, solve_time)."""
        start_time = time.time()
        solver_output = solver(problem)
        end_time = time.time()
        solve_time = end_time - start_time
        
        return solver_output, solve_time
    
    def _solve_validate_save(self, index: int, problem: MedicalNPProblem, solver: Callable) -> Dict:
        """Solve, validate, and save one problem in a worker thread."""
        print(f"[Worker] 开始处理问题 index={index}, name={problem.name}")
        
        try:
            # Step 1: solve
            start_time = time.time()
            solver_output = solver(problem)
            end_time = time.time()
            solve_time = end_time - start_time
            
            # Step 2: validate and process
            result = self._process_result(index, problem, solver_output, solve_time)
            
            print(f"[Worker] 完成问题 index={index}, name={problem.name}, Time={solve_time:.4f}s")
            return result
            
        except Exception as e:
            print(f"\n[Worker ERROR] 处理问题 {problem.name} 失败")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            
            error_result = {
                "index": index,
                "problem_name": problem.name,
                "problem_type": problem.problem_type,
                "complexity": problem.metadata.get("complexity", "unknown"),
                "solve_time": solve_time if 'solve_time' in locals() else 0,
                "valid": False,
                "violations": ["error"],
                "quality": 0,
                "metrics": {},
                "usage": {},
                "validation_details": {},
                "llm_raw_output": f"Error: {str(e)}",
                "llm_answer": None,
                "baseline": getattr(problem, 'baseline', None),
                "baseline_comparison": {},
                "error": str(e)
            }
            
            # Save error results in the worker thread
            if self.results_dir:
                try:
                    problem_name = problem.name
                    if "Diagnosis" in problem_name:
                        domain_dir = self.results_dir / "diagnosis"
                    elif "Pattern" in problem_name:
                        domain_dir = self.results_dir / "pattern_detection"
                    elif "Risk" in problem_name:
                        domain_dir = self.results_dir / "risk_assessment"
                    elif "Scheduling" in problem_name or "PASP" in problem_name:
                        domain_dir = self.results_dir / "scheduling"
                    else:
                        domain_dir = self.results_dir
                    
                    domain_dir.mkdir(parents=True, exist_ok=True)
                    result_filename = f"{index}_{problem_name.replace(' ', '_')}.json"
                    result_path = domain_dir / result_filename
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)
                    print(f"[Worker] 错误结果已保存到 {result_path}")
                except Exception as save_error:
                    print(f"[Worker] 警告: 保存错误结果失败: {save_error}")
            
            return error_result
    
    def _process_result(self, index: int, problem: MedicalNPProblem, solver_output: Dict, solve_time: float) -> Dict:
        """Process one solver result, including its elapsed solve time."""
        print(f"[_process_result] 处理问题 index={index}, name={problem.name}, type={problem.problem_type}")
        
        # Extract raw output and solution
        raw_output = solver_output.get("raw_output", "")
        solution = solver_output.get("solution", solver_output)  # Support the legacy format
        usage = solver_output.get("usage", {})  # Model usage information

        # Stop if the model returns an empty response, including reasoning-only output
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise EmptyLLMResponseError(
                f"Problem {problem.name} (index {index}) received empty LLM response. "
                f"Stopping benchmark run."
            )
        
        # Use the cached baseline
        baseline = getattr(problem, 'baseline', None)
        
        # Stop with an error if the baseline is missing
        if baseline is None:
            raise MissingBaselineError(f"Problem {problem.name} (index {index}) has no baseline. Please ensure baseline is generated before evaluation.")
        
        # Evaluate the solution
        evaluation_result = problem.validator.evaluate_solution(problem, solution, baseline)
        
        # Build the result
        problem_result = {
            "index": index,
            "problem_name": problem.name,
            "problem_type": problem.problem_type,
            "complexity": problem.metadata.get("complexity", "unknown"),
            "solve_time": solve_time,
            "valid": evaluation_result["valid"],
            "violations": evaluation_result["violations"],
            "quality": evaluation_result["score"],
            "metrics": {},
            "usage": usage,
            "validation_details": evaluation_result.get("details", {}),
            "llm_raw_output": raw_output,
            "llm_answer": solution,
            "baseline": baseline,
            "baseline_comparison": evaluation_result.get("baseline_comparison", {})
        }
        
        print(f"  Problem {index+1}: {problem.name} - Time: {solve_time:.4f}s, Valid: {evaluation_result['valid']}, Quality: {evaluation_result['score']:.4f}")
        
        # Save results during execution
        if self.results_dir:
            try:
                problem_name = problem.name
                if "Diagnosis" in problem_name:
                    domain_dir = self.results_dir / "diagnosis"
                elif "Pattern" in problem_name:
                    domain_dir = self.results_dir / "pattern_detection"
                elif "Risk" in problem_name:
                    domain_dir = self.results_dir / "risk_assessment"
                elif "Scheduling" in problem_name or "PASP" in problem_name:
                    domain_dir = self.results_dir / "scheduling"
                else:
                    domain_dir = self.results_dir
                
                domain_dir.mkdir(parents=True, exist_ok=True)
                
                result_filename = f"{index}_{problem_name.replace(' ', '_')}.json"
                result_path = domain_dir / result_filename
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(problem_result, f, indent=2, ensure_ascii=False, default=str)
                print(f"  [SAVE] 结果已保存到 {result_path}")
            except Exception as save_error:
                print(f"  警告: 保存结果失败: {save_error}")
        
        return problem_result
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics for a list of problem results."""
        if not results:
            return {
                "total_problems": 0,
                "valid_solutions": 0,
                "valid_rate": 0.0,
                "avg_solve_time": 0.0,
                "avg_quality": 0.0,
                "by_problem_type": {},
                "by_complexity": {}
            }
        
        total_problems = len(results)
        valid_solutions = sum(1 for r in results if r["valid"])
        avg_solve_time = sum(r["solve_time"] for r in results) / total_problems
        avg_quality = sum(r["quality"] for r in results) / total_problems
        
        # Summarize by problem type
        by_problem_type = {}
        by_complexity = {}
        
        for result in results:
            problem_type = result["problem_type"]
            complexity = result["complexity"]
            
            if problem_type not in by_problem_type:
                by_problem_type[problem_type] = {
                    "count": 0,
                    "valid": 0,
                    "total_time": 0,
                    "total_quality": 0
                }
            
            by_problem_type[problem_type]["count"] += 1
            by_problem_type[problem_type]["valid"] += 1 if result["valid"] else 0
            by_problem_type[problem_type]["total_time"] += result["solve_time"]
            by_problem_type[problem_type]["total_quality"] += result["quality"]
            
            if complexity not in by_complexity:
                by_complexity[complexity] = {
                    "count": 0,
                    "valid": 0,
                    "total_time": 0,
                    "total_quality": 0
                }
            
            by_complexity[complexity]["count"] += 1
            by_complexity[complexity]["valid"] += 1 if result["valid"] else 0
            by_complexity[complexity]["total_time"] += result["solve_time"]
            by_complexity[complexity]["total_quality"] += result["quality"]
        
        # Calculate average metrics by problem type
        for problem_type, stats in by_problem_type.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_quality"] = stats["total_quality"] / stats["count"]
            stats["valid_rate"] = stats["valid"] / stats["count"]
        
        # Calculate average metrics by complexity
        for complexity, stats in by_complexity.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_quality"] = stats["total_quality"] / stats["count"]
            stats["valid_rate"] = stats["valid"] / stats["count"]
        
        return {
            "total_problems": total_problems,
            "valid_solutions": valid_solutions,
            "valid_rate": valid_solutions / total_problems,
            "avg_solve_time": avg_solve_time,
            "avg_quality": avg_quality,
            "by_problem_type": by_problem_type,
            "by_complexity": by_complexity
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a benchmark report and optionally save it to output_file."""
        if not self.results:
            return "No benchmark results available."
        
        report = "# Medical NP Problem Benchmark Report\n\n"
        
        # Add summary statistics
        summary = self._calculate_summary(self.results)
        report += "## Summary\n\n"
        report += f"Total problems: {summary['total_problems']}\n"
        report += f"Valid solutions: {summary['valid_solutions']} ({summary['valid_rate']:.2%})\n"
        report += f"Average solve time: {summary['avg_solve_time']:.4f}s\n"
        report += f"Average quality: {summary['avg_quality']:.4f}\n\n"
        
        # Add statistics by problem type
        report += "## By Problem Type\n\n"
        for problem_type, stats in summary['by_problem_type'].items():
            report += f"### {problem_type}\n"
            report += f"  Count: {stats['count']}\n"
            report += f"  Valid: {stats['valid']} ({stats['valid_rate']:.2%})\n"
            report += f"  Avg time: {stats['avg_time']:.4f}s\n"
            report += f"  Avg quality: {stats['avg_quality']:.4f}\n\n"
        
        # Add statistics by complexity
        report += "## By Complexity\n\n"
        for complexity, stats in summary['by_complexity'].items():
            report += f"### {complexity}\n"
            report += f"  Count: {stats['count']}\n"
            report += f"  Valid: {stats['valid']} ({stats['valid_rate']:.2%})\n"
            report += f"  Avg time: {stats['avg_time']:.4f}s\n"
            report += f"  Avg quality: {stats['avg_quality']:.4f}\n\n"
        
        # Add detailed results
        report += "## Detailed Results\n\n"
        for i, result in enumerate(self.results):
            # Calculate evaluation grade
            valid = result['valid']
            quality = result['quality']
            
            if valid and quality >= 0.8:
                grade = "⭐ Excellent"
            elif valid and quality >= 0.6:
                grade = "✅ Good"
            elif valid and quality >= 0.4:
                grade = "⚠️ Fair"
            elif valid:
                grade = "🔶 Pass"
            else:
                grade = "❌ Failed"
            
            report += f"### Problem {i+1}: {result['problem_name']}\n"
            report += f"  Evaluation: {grade}\n"
            report += f"  Type: {result['problem_type']}\n"
            report += f"  Complexity: {result['complexity']}\n"
            report += f"  Solve time: {result['solve_time']:.4f}s\n"
            report += f"  Valid: {result['valid']}\n"
            report += f"  Quality: {result['quality']:.4f}\n"
            
            # Add detailed metrics such as coverage and probability scores
            validation_details = result.get('validation_details', {})
            if validation_details:
                # Coverage
                if 'coverage' in validation_details:
                    report += f"  Coverage: {validation_details['coverage']:.4f}\n"
                # Probability score
                if 'probability' in validation_details:
                    report += f"  Probability: {validation_details['probability']:.4f}\n"
                # Detailed score if different from quality
                if 'score' in validation_details and validation_details['score'] != result['quality']:
                    report += f"  Detailed score: {validation_details['score']:.4f}\n"
                # Other details
                if 'unused_time' in validation_details and validation_details['unused_time'] is not None:
                    report += f"  Unused time: {validation_details['unused_time']:.2f} minutes\n"
                if 'cancellations' in validation_details:
                    report += f"  Cancellations: {validation_details['cancellations']}\n"
                # Correctness/accuracy
                if 'accuracy' in validation_details:
                    report += f"  Accuracy: {validation_details['accuracy']:.4f}\n"
            
            if result['violations']:
                report += f"  Violations: {', '.join(result['violations'])}\n"
            
            # Add baseline comparison details
            baseline_comparison = result.get('baseline_comparison', {})
            if baseline_comparison:
                if 'baseline_score' in baseline_comparison:
                    report += f"  Baseline score: {baseline_comparison['baseline_score']:.4f}\n"
                if 'llm_score' in baseline_comparison:
                    report += f"  LLM score: {baseline_comparison['llm_score']:.4f}\n"
                if 'llm_better' in baseline_comparison:
                    report += f"  LLM better than baseline: {baseline_comparison['llm_better']}\n"
                # Compare answers for Yes/No tasks
                if 'baseline_answer' in baseline_comparison and 'llm_answer' in baseline_comparison:
                    report += f"  Baseline answer: {baseline_comparison['baseline_answer']}\n"
                    report += f"  LLM answer: {baseline_comparison['llm_answer']}\n"
                    if 'correct' in baseline_comparison:
                        report += f"  Answer correct: {baseline_comparison['correct']}\n"
            
            report += "\n"
        
        # Save the report to a file
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report
