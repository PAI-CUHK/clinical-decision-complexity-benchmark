import re
from typing import Optional


def normalize_yes_no(answer) -> str:
    """Normalize an answer to yes/no for comparison."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    s = str(answer).strip().lower()
    if s in ("yes", "true", "1", "是"):
        return "yes"
    return "no"


def extract_yes_no_answer(output) -> Optional[str]:
    """Extract a Yes/No answer from the final_answer tag, preserving its original
    capitalization. Return None when the tag is missing, including when the output
    is truncated or incorrectly formatted."""
    if isinstance(output, bool):
        return "Yes" if output else "No"

    if not isinstance(output, str):
        return None

    text = output
    root_match = re.search(r"<root>\s*([\s\S]*?)\s*</root>", text, re.DOTALL | re.IGNORECASE)
    if root_match:
        text = root_match.group(1)

    match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Return None when no final_answer tag is found
    return None
