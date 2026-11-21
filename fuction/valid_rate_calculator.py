import re
from collections import defaultdict
from typing import Dict, List, Any


def get_each_problem_valid_rate(report_path: str) -> List[Dict[str, Any]]:
    """Read each problem ID, name, and valid rate from the report path."""
    results = []
    
    with open(report_path, 'r') as f:
        lines = f.readlines()
    
    problem_pattern = re.compile(r'### Problem (\d+): (.+)')
    current_id = None
    current_name = None
    current_valid = None
    
    for line in lines:
        line = line.strip()
        
        problem_match = problem_pattern.match(line)
        if problem_match:
            if current_id is not None:
                valid_rate = 100.0 if current_valid else 0.0
                results.append({
                    'problem_id': current_id,
                    'problem_name': current_name,
                    'valid': current_valid,
                    'valid_rate': valid_rate
                })
            current_id = int(problem_match.group(1))
            current_name = problem_match.group(2)
            current_valid = None
            continue
        
        if line.startswith('Valid:'):
            current_valid = line.split(':')[1].strip() == 'True'
    
    if current_id is not None:
        valid_rate = 100.0 if current_valid else 0.0
        results.append({
            'problem_id': current_id,
            'problem_name': current_name,
            'valid': current_valid,
            'valid_rate': valid_rate
        })
    
    return results


def calculate_valid_rate_by_problem_name(report_path: str) -> Dict[str, Dict[str, Any]]:
    """Calculate valid-rate statistics by problem name from the report path."""
    stats = defaultdict(lambda: {'total': 0, 'valid': 0})
    
    with open(report_path, 'r') as f:
        lines = f.readlines()
    
    problem_pattern = re.compile(r'### Problem (\d+): (.+)')
    current_name = None
    current_valid = None
    
    for line in lines:
        line = line.strip()
        
        problem_match = problem_pattern.match(line)
        if problem_match:
            if current_name is not None:
                base_name = current_name.split(' Level')[0] if ' Level' in current_name else current_name
                stats[base_name]['total'] += 1
                if current_valid:
                    stats[base_name]['valid'] += 1
            current_name = problem_match.group(2)
            current_valid = None
            continue
        
        if line.startswith('Valid:'):
            current_valid = line.split(':')[1].strip() == 'True'
    
    if current_name is not None:
        base_name = current_name.split(' Level')[0] if ' Level' in current_name else current_name
        stats[base_name]['total'] += 1
        if current_valid:
            stats[base_name]['valid'] += 1
    
    result = {}
    for problem_name, data in stats.items():
        rate = (data['valid'] / data['total']) * 100 if data['total'] > 0 else 0
        result[problem_name] = {
            'total': data['total'],
            'valid': data['valid'],
            'valid_rate': round(rate, 2)
        }
    
    return result


def parse_report_file(report_path: str) -> List[Dict[str, Any]]:
    """Parse problem details from the report file and return a list."""
    problems = []
    current_problem = {}
    
    with open(report_path, 'r') as f:
        lines = f.readlines()
    
    problem_pattern = re.compile(r'### Problem (\d+): (.+)')
    
    for line in lines:
        line = line.strip()
        
        problem_match = problem_pattern.match(line)
        if problem_match:
            if current_problem:
                problems.append(current_problem)
            current_problem = {
                'id': int(problem_match.group(1)),
                'name': problem_match.group(2),
                'level': extract_level(problem_match.group(2))
            }
            continue
        
        if line.startswith('Type:'):
            current_problem['type'] = line.split(':')[1].strip()
        elif line.startswith('Complexity:'):
            current_problem['complexity'] = line.split(':')[1].strip()
        elif line.startswith('Valid:'):
            current_problem['valid'] = line.split(':')[1].strip() == 'True'
    
    if current_problem:
        problems.append(current_problem)
    
    return problems


def extract_level(name: str) -> int:
    """Extract the numeric scale level from a problem name."""
    match = re.search(r'Level (\d+)', name)
    return int(match.group(1)) if match else 0


def calculate_valid_rate_by_type(problems: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate valid-rate statistics by problem type."""
    stats = defaultdict(lambda: {'total': 0, 'valid': 0})
    
    for problem in problems:
        problem_type = problem.get('type', 'unknown')
        stats[problem_type]['total'] += 1
        if problem.get('valid', False):
            stats[problem_type]['valid'] += 1
    
    result = {}
    for problem_type, data in stats.items():
        rate = (data['valid'] / data['total']) * 100 if data['total'] > 0 else 0
        result[problem_type] = {
            'total': data['total'],
            'valid': data['valid'],
            'valid_rate': round(rate, 2)
        }
    
    return result


def calculate_valid_rate_by_complexity(problems: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate valid-rate statistics by complexity class."""
    stats = defaultdict(lambda: {'total': 0, 'valid': 0})
    
    for problem in problems:
        complexity = problem.get('complexity', 'unknown')
        stats[complexity]['total'] += 1
        if problem.get('valid', False):
            stats[complexity]['valid'] += 1
    
    result = {}
    for complexity, data in stats.items():
        rate = (data['valid'] / data['total']) * 100 if data['total'] > 0 else 0
        result[complexity] = {
            'total': data['total'],
            'valid': data['valid'],
            'valid_rate': round(rate, 2)
        }
    
    return result


def calculate_valid_rate_by_level(problems: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate valid-rate statistics by level."""
    stats = defaultdict(lambda: {'total': 0, 'valid': 0})
    
    for problem in problems:
        level = problem.get('level', 0)
        stats[level]['total'] += 1
        if problem.get('valid', False):
            stats[level]['valid'] += 1
    
    result = {}
    for level in sorted(stats.keys()):
        data = stats[level]
        rate = (data['valid'] / data['total']) * 100 if data['total'] > 0 else 0
        result[f'Level {level}'] = {
            'total': data['total'],
            'valid': data['valid'],
            'valid_rate': round(rate, 2)
        }
    
    return result


def calculate_valid_rate_by_type_and_level(problems: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Calculate valid-rate statistics by problem type and level."""
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'valid': 0}))
    
    for problem in problems:
        problem_type = problem.get('type', 'unknown')
        level = problem.get('level', 0)
        stats[problem_type][level]['total'] += 1
        if problem.get('valid', False):
            stats[problem_type][level]['valid'] += 1
    
    result = {}
    for problem_type, levels in stats.items():
        result[problem_type] = {}
        for level in sorted(levels.keys()):
            data = levels[level]
            rate = (data['valid'] / data['total']) * 100 if data['total'] > 0 else 0
            result[problem_type][f'Level {level}'] = {
                'total': data['total'],
                'valid': data['valid'],
                'valid_rate': round(rate, 2)
            }
    
    return result


def get_overall_valid_rate(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall valid-rate statistics."""
    total = len(problems)
    valid = sum(1 for p in problems if p.get('valid', False))
    rate = (valid / total) * 100 if total > 0 else 0
    
    return {
        'total': total,
        'valid': valid,
        'valid_rate': round(rate, 2)
    }


def generate_valid_rate_report(report_path: str) -> str:
    """Generate a valid-rate report string from the report file."""
    problems = parse_report_file(report_path)
    each_problem = get_each_problem_valid_rate(report_path)
    by_problem_name = calculate_valid_rate_by_problem_name(report_path)
    
    overall = get_overall_valid_rate(problems)
    by_type = calculate_valid_rate_by_type(problems)
    by_complexity = calculate_valid_rate_by_complexity(problems)
    by_level = calculate_valid_rate_by_level(problems)
    by_type_level = calculate_valid_rate_by_type_and_level(problems)
    
    report = []
    report.append("# Valid Rate Report")
    report.append("")
    report.append("## Overall")
    report.append(f"  Total problems: {overall['total']}")
    report.append(f"  Valid solutions: {overall['valid']} ({overall['valid_rate']}%)")
    report.append("")
    report.append("## Each Problem Valid Rate")
    for problem in each_problem:
        report.append(f"### Problem {problem['problem_id']}: {problem['problem_name']}")
        report.append(f"  Valid: {'Yes' if problem['valid'] else 'No'} ({problem['valid_rate']}%)")
    report.append("")
    report.append("## By Problem Name")
    for problem_name, stats in by_problem_name.items():
        report.append(f"### {problem_name}")
        report.append(f"  Count: {stats['total']}")
        report.append(f"  Valid: {stats['valid']} ({stats['valid_rate']}%)")
    report.append("")
    report.append("## By Problem Type")
    for problem_type, stats in by_type.items():
        report.append(f"### {problem_type}")
        report.append(f"  Count: {stats['total']}")
        report.append(f"  Valid: {stats['valid']} ({stats['valid_rate']}%)")
    report.append("")
    report.append("## By Complexity")
    for complexity, stats in by_complexity.items():
        report.append(f"### {complexity}")
        report.append(f"  Count: {stats['total']}")
        report.append(f"  Valid: {stats['valid']} ({stats['valid_rate']}%)")
    report.append("")
    report.append("## By Level")
    for level, stats in by_level.items():
        report.append(f"### {level}")
        report.append(f"  Count: {stats['total']}")
        report.append(f"  Valid: {stats['valid']} ({stats['valid_rate']}%)")
    report.append("")
    report.append("## By Type and Level")
    for problem_type, levels in by_type_level.items():
        report.append(f"### {problem_type}")
        for level, stats in levels.items():
            report.append(f"  {level}: {stats['valid']}/{stats['total']} ({stats['valid_rate']}%)")
    
    return '\n'.join(report)


def calculate_valid_rates(report_path: str) -> Dict[str, Any]:
    """Return structured valid-rate statistics across all dimensions."""
    problems = parse_report_file(report_path)
    
    return {
        'overall': get_overall_valid_rate(problems),
        'by_type': calculate_valid_rate_by_type(problems),
        'by_complexity': calculate_valid_rate_by_complexity(problems),
        'by_level': calculate_valid_rate_by_level(problems),
        'by_type_and_level': calculate_valid_rate_by_type_and_level(problems),
        'detailed': problems
    }