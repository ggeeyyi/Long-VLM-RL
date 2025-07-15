# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict

from mathruler.grader import grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response.replace("\n", "").replace(" ", ""))
        given_answer = content_match.group(1).strip() if content_match else response.strip()
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0

    except Exception:
        pass

    return 0.0


def sqa_accuracy_reward(response: str, ground_truth: str) -> float:
    """
    SQA-style accuracy reward function based on token matching.
    Splits response by spaces, periods, and commas, then checks if ground_truth matches any token using grade_answer.
    Requires both <answer> and </answer> tags to be present.
    """
    try:
        # Extract answer from response - must have both <answer> and </answer> tags
        content_match = re.search(r"<answer>(.*?)</answer>", response.replace("\n", "").replace(" ", ""))
        if not content_match:
            return 0.0
        
        pred = content_match.group(1).strip()
        
        # Normalize target
        target = ground_truth.strip()
        
        # Split response by spaces, periods, and commas
        tokens = re.split(r'[ .,:;]+', pred)
        
        # Remove empty tokens and strip whitespace
        tokens = [token.strip() for token in tokens if token.strip()]
        
        # Check if target matches any token using grade_answer
        for token in tokens:
            if grade_answer(token, target):
                return 1.0
        
        return 0.0
        
    except Exception:
        return 0.0


def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")

    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


def sqa_compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    """
    SQA-style compute score function using SQA accuracy logic.
    """
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for SQA reward function.")

    format_score = format_reward(reward_input["response"])
    accuracy_score = sqa_accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
