"""
Simplified reward computation for math problems
Focuses on core mathematical equivalence checking without verl dependencies
"""

import re
from enum import Enum, auto
from typing import NamedTuple


class Status(Enum):
    OK = auto()
    NO_INP = auto()
    NOT_EQUIV = auto()
    ERROR = auto()


class Score(NamedTuple):
    result: float
    report: str
    status: Status


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last boxed answer from a string."""
    idx = string.rfind("\\boxed")
    if idx == -1:
        idx = string.rfind("\\fbox")
        if idx == -1:
            return None
    
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s: str) -> str:
    """Remove the boxed wrapper from a string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer string for comparison."""
    final_answer = final_answer.strip()
    if final_answer.endswith("."):
        final_answer = final_answer[:-1]
    
    # Remove common formatting
    final_answer = final_answer.replace("$", "")
    final_answer = final_answer.replace(",", "")
    final_answer = final_answer.strip()
    
    # Handle fractions
    if "/" in final_answer:
        try:
            parts = final_answer.split("/")
            if len(parts) == 2:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den != 0:
                    final_answer = str(num / den)
        except:
            pass
    
    return final_answer


def is_equiv(str1: str, str2: str) -> bool:
    """Check if two strings represent equivalent mathematical expressions."""
    if str1 is None or str2 is None:
        return False
    
    str1 = str1.strip()
    str2 = str2.strip()
    
    # Direct string comparison
    if str1 == str2:
        return True
    
    # Normalize and compare
    norm1 = normalize_final_answer(str1)
    norm2 = normalize_final_answer(str2)
    
    if norm1 == norm2:
        return True
    
    # Try numerical comparison
    try:
        float1 = float(norm1)
        float2 = float(norm2)
        return abs(float1 - float2) < 1e-6
    except:
        pass
    
    # Try integer comparison
    try:
        int1 = int(norm1)
        int2 = int(norm2)
        return int1 == int2
    except:
        pass
    
    # Handle common mathematical notations
    # Remove spaces
    str1_clean = re.sub(r'\s+', '', str1.lower())
    str2_clean = re.sub(r'\s+', '', str2.lower())
    
    if str1_clean == str2_clean:
        return True
    
    return False


def compute_score(solution_str: str, ground_truth: str) -> Score:
    """Compute the score for a solution against ground truth."""
    try:
        # Extract the last boxed answer
        string_in_last_boxed = last_boxed_only_string(solution_str)
        
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                return Score(1.0, solution_str, Status.OK)
            else:
                return Score(0.0, solution_str, Status.NOT_EQUIV)
        else:
            # Try to find answer patterns without boxed notation
            # Look for common answer patterns
            patterns = [
                r'(?i)(?:answer|final answer|solution)(?:\s*[:=]\s*)(.+)',
                r'(?i)(?:therefore|thus|so)(?:\s*,?\s*)(.+)',
                r'(?i)the answer is\s*(.+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, solution_str)
                if match:
                    candidate_answer = match.group(1).strip()
                    # Clean up the candidate answer
                    candidate_answer = re.sub(r'[.!,]+$', '', candidate_answer)
                    if is_equiv(candidate_answer, ground_truth):
                        return Score(1.0, solution_str, Status.OK)
            
            return Score(0.0, solution_str, Status.NO_INP)
            
    except Exception as e:
        return Score(0.0, solution_str + "\n" + str(e), Status.ERROR)