"""
ECG RR-Interval Pattern Recognition Adapter

Translates an ECG RR-interval pattern recognition instance into the unified
problem framework.
"""

from core import MedicalProblemAdapter, MedicalNPProblem, Action, Constraint, Objective


class ECGPatternAdapter(MedicalProblemAdapter):
    """
    ECG RR-Interval Pattern Recognition Adapter.

    This is a P-class pattern matching problem:
    - Input: ordered sequence of RR intervals (seconds)
    - Task: detect four predefined RR-interval screening patterns
    - Complexity: O(n) linear scan with a rolling median baseline

    Patterns:
    Pattern 1 - Short-Long RR Pattern
        A Short RR interval immediately followed by a Long RR interval.
        Rule: RR_i is Short and RR_{i+1} is Long.

    Pattern 2 - Periodic Short-Long RR Pattern
        A repeated Short-Long RR timing pattern with constant periodicity
        that repeats at least three consecutive times.

    Pattern 3 - Prolonged RR Interval
        A prolonged RR interval meeting the pause criterion.
        Rule: RR_i >= 2.0 seconds.

    Pattern 4 - Relative RR Difference Episode
        A sliding window of 10 consecutive RR intervals containing at least
        8 Relative RR Difference Events (relative difference >= 3.25%).

    RR classification (from RR_6 onward):
    - Baseline RR for RR_i = median of RR_{i-5} .. RR_{i-1}
    - Short : RR <  80% of Baseline RR
    - Long  : RR > 120% of Baseline RR
    - Normal: neither Short nor Long
    """

    def generate_instance_config(self, level):
        """
        Generate the ECG instance configuration for a given difficulty level.

        Args:
            level: difficulty level (1-based)

        Returns:
            dict: instance configuration
        """
        num_intervals = 24 + level * 4
        return {
            "num_intervals": num_intervals,
            "baseline_rr": 0.9,
            "max_rr": 2.6,
        }

    def build(self, instance):
        problem = MedicalNPProblem(
            name="ECG RR Interval Pattern Detection",
            problem_type="pattern_detection"
        )

        problem.instance_data = {
            "rr_intervals": instance.rr_intervals
        }

        problem.description = (
            "Given a sequence of RR intervals, identify all occurrences of predefined "
            "RR-interval screening patterns. "
            "All RR interval data are known in advance. "
            "Return the exact 1-based starting positions for each detected pattern."
        )

        problem.clinical_context = {
            "description": (
                "RR intervals represent the time between consecutive heartbeats. "
                "RR interval analysis is commonly used during ambulatory ECG (Holter) "
                "monitoring to screen for temporal rhythm patterns. "
                "This task is limited to RR-interval screening only. "
                "Do not infer or report any cardiac rhythm diagnosis."
            )
        }

        problem.decision_space = [
            "Scan the RR interval sequence and identify every occurrence of each predefined pattern.",
            "For Pattern 1-3, report the starting RR interval of the detected pattern.",
            "For Pattern 4, report the starting RR interval of each qualifying sliding window.",
            "Use 1-based indexing and report all occurrences, including overlapping ones."
        ]

        num_intervals = len(instance.rr_intervals)
        problem.metadata = {
            "num_rr_intervals": num_intervals,
            "complexity": "P",
            "problem_type": "pattern_detection"
        }

        for i, rr in enumerate(instance.rr_intervals):
            problem.add_entity("rr_intervals", {
                "index": i + 1,
                "value": rr
            })

        for pattern_name in ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4"]:
            for i in range(num_intervals):
                problem.add_action(Action(
                    name=f"detect_{pattern_name.replace(' ', '_')}_at_{i + 1}",
                    description=f"Detect {pattern_name} starting at position {i + 1}"
                ))

        # ========= Definitions and Rules =========
        # --- RR Classification ---
        problem.add_constraint(Constraint(
            name="baseline_rr_definition",
            description=(
                "Baseline RR: Beginning with RR_6, the Baseline RR for RR_i is the median of "
                "the five immediately preceding RR intervals (RR_{i-5} to RR_{i-1}). "
                "The first five RR intervals are reference intervals only."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "baseline", "scope": "window_5_median"}
        ))
        problem.add_constraint(Constraint(
            name="short_rr_definition",
            description=(
                "Short RR: An RR interval is Short if its duration is less than 80% of its "
                "corresponding Baseline RR."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "classification", "label": "Short", "ratio": 0.80, "direction": "below"}
        ))
        problem.add_constraint(Constraint(
            name="long_rr_definition",
            description=(
                "Long RR: An RR interval is Long if its duration is greater than 120% of its "
                "corresponding Baseline RR."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "classification", "label": "Long", "ratio": 1.20, "direction": "above"}
        ))
        problem.add_constraint(Constraint(
            name="normal_rr_definition",
            description=(
                "Normal RR: An RR interval is Normal if it is neither Short nor Long."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "classification", "label": "Normal"}
        ))

        # --- Pattern Definitions ---
        problem.add_constraint(Constraint(
            name="pattern_1_definition",
            description=(
                "Pattern 1 - Short-Long RR Pattern. "
                "Definition: A Short RR interval immediately followed by a Long RR interval. "
                "Rule: Pattern 1 occurs if RR_i is Short and RR_{i+1} is Long. "
                "Report the position of RR_i."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "pattern_definition", "pattern": "Pattern 1", "scope": "pair"}
        ))
        problem.add_constraint(Constraint(
            name="pattern_2_definition",
            description=(
                "Pattern 2 - Periodic Short-Long RR Pattern. "
                "Definition: A repeated Short-Long RR timing pattern with constant periodicity. "
                "Rule: Pattern 2 occurs if every repeated unit consists of one Short RR "
                "immediately followed by one Long RR; the number of consecutive Normal RR "
                "intervals between neighboring Short RR intervals remains constant; and the "
                "same pattern repeats at least three consecutive times. "
                "Report the 1-based position of the first Short RR interval of each qualifying "
                "periodic chain. A periodic chain is a maximal run of consecutive Short-Long "
                "units sharing the same constant Normal-gap. Even if a chain contains more than "
                "three units, report only the first Short RR of that chain — do NOT report "
                "sliding-window sub-sequences."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "pattern_definition", "pattern": "Pattern 2", "scope": "periodic",
                       "min_repeats": 3}
        ))
        problem.add_constraint(Constraint(
            name="pattern_3_definition",
            description=(
                "Pattern 3 - Prolonged RR Interval. "
                "Definition: A prolonged RR interval meeting the predefined pause criterion. "
                "Rule: Pattern 3 occurs if RR_i >= 2.0 seconds. "
                "Report the position of RR_i."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "pattern_definition", "pattern": "Pattern 3", "scope": "single",
                       "threshold": 2.0}
        ))
        problem.add_constraint(Constraint(
            name="pattern_4_definition",
            description=(
                "Pattern 4 - Relative RR Difference Episode. "
                "Definition: A local episode characterized by frequent large relative "
                "differences between successive RR intervals. "
                "Rule: For every adjacent RR pair, compute |RR_{i+1} - RR_i| / RR_i; if this "
                "relative difference is at least 3.25%, record one Relative RR Difference "
                "Event. Then examine every sliding window of 10 consecutive RR intervals; if "
                "a window contains at least 8 events, Pattern 4 is detected. "
                "Report the starting position of the sliding window."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "pattern_definition", "pattern": "Pattern 4", "scope": "sliding_window",
                       "window": 10, "min_events": 8, "rel_diff_threshold": 0.0325}
        ))

        # ========= Validity Constraints =========
        problem.add_constraint(Constraint(
            name="validity_use_only_provided_data",
            description="Use only the RR interval sequence provided.",
            level="clinical", weight=0.0, category="validity"
        ))
        problem.add_constraint(Constraint(
            name="validity_apply_rules_exactly",
            description="Apply the predefined rules exactly.",
            level="clinical", weight=0.0, category="validity"
        ))
        problem.add_constraint(Constraint(
            name="validity_1based_indexing",
            description="Use 1-based indexing.",
            level="clinical", weight=0.0, category="validity"
        ))
        problem.add_constraint(Constraint(
            name="validity_report_every_occurrence",
            description="Report every occurrence of every pattern.",
            level="clinical", weight=0.0, category="validity"
        ))
        problem.add_constraint(Constraint(
            name="validity_no_omitted_overlapping",
            description="Do not omit overlapping occurrences.",
            level="clinical", weight=0.0, category="validity"
        ))
        problem.add_constraint(Constraint(
            name="validity_empty_array_if_no_occurrence",
            description="If a pattern does not occur, return an empty array for that pattern.",
            level="clinical", weight=0.0, category="validity"
        ))
        problem.add_constraint(Constraint(
            name="validity_no_diagnosis",
            description="Do not infer or report any cardiac rhythm diagnosis.",
            level="clinical", weight=0.0, category="validity"
        ))

        # ========= Task Objective =========
        problem.add_objective(Objective(
            name="detect_all_patterns_correctly",
            description=(
                "Identify every occurrence of the four predefined RR-interval screening "
                "patterns with no false positives and no false negatives."
            ),
            direction="maximize", weight=1.0, priority=1,
            semantics={"type": "accuracy", "scope": "all_patterns"}
        ))

        return problem
