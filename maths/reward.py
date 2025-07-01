import json
import os
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from verl import DataProto
from verl.utils.reward_score.math import is_equiv
from verl.utils.reward_score.math import last_boxed_only_string
from verl.utils.reward_score.math import remove_boxed
from verl.workers.reward_manager.self_reward import decode_batch


class Status(Enum):
    OK = auto()
    NO_INP = auto()
    NOT_EQUIV = auto()
    ERROR = auto()


class Score(NamedTuple):
    result: float | int
    report: str
    status: Status


def compute_score(solution_str: str, ground_truth: str) -> Score:
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                return Score(1.0, solution_str, Status.OK)
            else:
                return Score(0.0, solution_str, Status.NOT_EQUIV)
        else:
            return Score(0.0, solution_str, Status.NO_INP)
    except Exception as e:
        return Score(0.0, solution_str + "\n" + str(e), Status.ERROR)


@dataclass
class ResultsLogger:
    logs_dir: str
    logs_prefix: str

    def log_results(
        self,
        extra_infos: List[Dict[str, Any]],
        reward_models: List[Dict[str, Any]],
        results: List[Score],
        eval_step: int,
    ):
        data = []
        for extra_info, reward_model, result in zip(
            extra_infos, reward_models, results
        ):
            # TODO: It assumes non nested dicts for extra info ?
            datum = extra_info.copy() | reward_model.copy()
            datum["report"] = result.report
            datum["status"] = result.status.name
            datum["result"] = result.result
            data.append(datum)

        file_name = f"{self.logs_prefix}_{eval_step}.json"
        with open(os.path.join(self.logs_dir, file_name), "w") as file:
            json.dump(data, file)


class MathRewardManager:
    """Custom reward manager for Countdown task that uses LLM judges."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        val_logger: ResultsLogger | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.val_logger = val_logger

        # HACK: I need to keep track at which step I'm for the logs.
        # Unless I modify more our fork of verl, I don't know how to
        # retrieve at which step I'm. RewardManager should be only executed
        # by one actor. No concurrency issues are taking into account yet.
        if self.val_logger:
            self._eval_step = 0
        else:
            self._eval_step = None

    def verify_batch(self, data: DataProto) -> List[Score]:
        responses = decode_batch(data, self.tokenizer)
        targets: List[str] = [
            item.non_tensor_batch["reward_model"]["ground_truth"] for item in data
        ]

        results: List[Score] = []
        for response, target in zip(responses, targets):
            results.append(compute_score(solution_str=response, ground_truth=target))

        return results

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Almost the same code as from BatchRewardManager from Verl.
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, List[int] | List[float]] = {
            status_code.name: [0] * len(data) for status_code in Status
        }

        # HACK: Verl requires all extra info tensors to be of same
        # lenght as the size of the batch. To not mess more with the fork
        # we instead use nanmean, and nansum, which computes those
        # collectives ignoring the nans.
        reward_extra_info["OK_LENGTH"] = [np.nan] * len(data)
        reward_extra_info["NO_OK_LENGTH"] = [np.nan] * len(data)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        scores = self.verify_batch(data)

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            result = scores[i].result
            status = scores[i].status

            reward_tensor[i, length - 1] = result
            reward_extra_info[status.name][i] = 1

            match status:
                case Status.OK:
                    reward_extra_info["OK_LENGTH"][i] = length
                case Status.NO_INP | Status.NOT_EQUIV | Status.ERROR:
                    reward_extra_info["NO_OK_LENGTH"][i] = length

        if self.val_logger:
            self.val_logger.log_results(
                data.non_tensor_batch["extra_info"],
                data.non_tensor_batch["reward_model"],
                scores,
                self._eval_step,
            )
            self._eval_step += 1

        if return_dict:
            return {  # pyright: ignore
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
