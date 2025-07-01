import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from tqdm import tqdm

from maths.reward_simple import compute_score, Score


@dataclass
class BenchmarkConfig:
    model_name: str
    revision: Optional[str] = None
    max_questions: int = 500
    batch_size: int = 512
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.95
    device_map: str = "auto"
    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    output_dir: str = "./benchmark_results"
    dataset_name: str = "DigitalLearningGmbH/MATH-lighteval"


class OLMo2Benchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.setup_logging()
        self.setup_distributed()
        self.load_model_and_tokenizer()
        self.load_dataset()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_distributed(self):
        if torch.cuda.device_count() > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            self.is_distributed = True
            self.logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_distributed = False
            
    def load_model_and_tokenizer(self):
        self.logger.info(f"Loading model: {self.config.model_name}")
        if self.config.revision:
            self.logger.info(f"Using revision: {self.config.revision}")
            
        torch_dtype = getattr(torch, self.config.torch_dtype)
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device_map if not self.is_distributed else None,
            "trust_remote_code": True,
        }
        
        if self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            
        if self.config.revision:
            model_kwargs["revision"] = self.config.revision
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.is_distributed:
            self.model = self.model.to(f'cuda:{self.local_rank}')
            self.model = DDP(self.model, device_ids=[self.local_rank])
            
        self.model.eval()
        
    def load_dataset(self):
        self.logger.info(f"Loading dataset: {self.config.dataset_name}")
        dataset = datasets.load_dataset(self.config.dataset_name, trust_remote_code=True)
        self.test_dataset = dataset["test"]
        
        if self.config.max_questions < len(self.test_dataset):
            indices = list(range(self.config.max_questions))
            self.test_dataset = self.test_dataset.select(indices)
            
        self.logger.info(f"Using {len(self.test_dataset)} questions for benchmarking")
        
    def prepare_prompts(self, problems: List[str]) -> List[str]:
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        return [f"{problem} {instruction}" for problem in problems]
        
    def generate_batch(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if self.is_distributed or torch.cuda.is_available():
            device = f'cuda:{self.local_rank}' if self.is_distributed else 'cuda'
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
            
        with torch.no_grad():
            model = self.model.module if self.is_distributed else self.model
            outputs = model.generate(**inputs, **generation_kwargs)
            
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        responses = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        return responses
        
    def run_benchmark(self) -> Dict[str, Any]:
        self.logger.info("Starting benchmark...")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        results = []
        correct_count = 0
        total_time = 0
        
        num_batches = (len(self.test_dataset) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in tqdm(range(0, len(self.test_dataset), self.config.batch_size), 
                     desc="Processing batches", disable=self.rank != 0):
            batch_end = min(i + self.config.batch_size, len(self.test_dataset))
            batch = self.test_dataset[i:batch_end]
            batch_num = (i // self.config.batch_size) + 1
            
            if self.rank == 0:
                self.logger.info(f"Processing batch {batch_num}/{num_batches} (questions {i+1}-{batch_end})")
            
            problems = batch["problem"]
            solutions = batch["solution"]
            
            prompts = self.prepare_prompts(problems)
            
            start_time = time.time()
            responses = self.generate_batch(prompts)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            if self.rank == 0:
                avg_time = batch_time / len(responses)
                self.logger.info(f"Batch {batch_num} completed in {batch_time:.1f}s ({avg_time:.2f}s per question)")
            
            for j, (problem, solution, response) in enumerate(zip(problems, solutions, responses)):
                ground_truth = self.extract_ground_truth(solution)
                score = compute_score(response, ground_truth)
                
                if score.result > 0:
                    correct_count += 1
                    
                result = {
                    "index": i + j,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "response": response,
                    "score": score.result,
                    "status": score.status.name,
                    "batch_time": batch_time / len(responses)
                }
                results.append(result)
                
        accuracy = correct_count / len(results) if results else 0
        avg_time_per_question = total_time / len(results) if results else 0
        
        benchmark_summary = {
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "total_questions": len(results),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "total_time": total_time,
            "avg_time_per_question": avg_time_per_question,
            "config": self.config.__dict__
        }
        
        if self.rank == 0:
            self.save_results(results, benchmark_summary)
            
        return benchmark_summary
        
    def extract_ground_truth(self, solution: str) -> str:
        from maths.math_dataset import extract_solution
        return extract_solution(solution)
        
    def save_results(self, results: List[Dict], summary: Dict[str, Any]):
        timestamp = int(time.time())
        model_name_clean = self.config.model_name.replace("/", "_")
        revision_suffix = f"_{self.config.revision}" if self.config.revision else ""
        
        results_file = f"{model_name_clean}{revision_suffix}_results_{timestamp}.json"
        summary_file = f"{model_name_clean}{revision_suffix}_summary_{timestamp}.json"
        
        results_path = os.path.join(self.config.output_dir, results_file)
        summary_path = os.path.join(self.config.output_dir, summary_file)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info(f"Summary saved to: {summary_path}")
        self.logger.info(f"Accuracy: {summary['accuracy']:.4f}")
        self.logger.info(f"Average time per question: {summary['avg_time_per_question']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OLMo 2 models on MATH dataset")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name (e.g., allenai/OLMo-2-0425-1B)")
    parser.add_argument("--revision", type=str, default=None,
                       help="Model revision/checkpoint (e.g., stage1-step140000-tokens294B)")
    parser.add_argument("--max_questions", type=int, default=500,
                       help="Maximum number of questions to test")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for sampling (0.0 for greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="PyTorch dtype for model")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit precision")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--dataset_name", type=str, default="DigitalLearningGmbH/MATH-lighteval",
                       help="Dataset name on HuggingFace Hub")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_name=args.model_name,
        revision=args.revision,
        max_questions=args.max_questions,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        load_in_8bit=args.load_in_8bit,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
    
    benchmark = OLMo2Benchmark(config)
    summary = benchmark.run_benchmark()
    
    print(f"\nBenchmark completed!")
    print(f"Model: {config.model_name}")
    if config.revision:
        print(f"Revision: {config.revision}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Correct: {summary['correct_answers']}/{summary['total_questions']}")
    print(f"Average time per question: {summary['avg_time_per_question']:.2f}s")


if __name__ == "__main__":
    main()