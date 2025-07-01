#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    """
    Example benchmarking runs for OLMo 2 models
    """
    
    examples = [
        {
            "name": "OLMo 2 1B Base Model",
            "cmd": "./run_benchmark.sh allenai/OLMo-2-0425-1B",
            "description": "Benchmark the base 1B model with default settings"
        },
        {
            "name": "OLMo 2 1B with Specific Checkpoint",
            "cmd": "./run_benchmark.sh allenai/OLMo-2-0425-1B stage1-step140000-tokens294B",
            "description": "Benchmark a specific checkpoint of the 1B model"
        },
        {
            "name": "OLMo 2 1B Instruct Model",
            "cmd": "./run_benchmark.sh allenai/OLMo-2-0425-1B-Instruct '' --batch_size 16 --max_questions 1000",
            "description": "Benchmark the instruct variant with larger batch size and more questions"
        },
        {
            "name": "OLMo 2 7B Base Model (Smaller Batch)",
            "cmd": "./run_benchmark.sh allenai/OLMo-2-1124-7B '' --batch_size 4 --load_in_8bit",
            "description": "Benchmark 7B model with 8-bit quantization for memory efficiency"
        },
        {
            "name": "OLMo 2 1B with Custom Settings",
            "cmd": "./run_benchmark.sh allenai/OLMo-2-0425-1B '' --temperature 0.7 --top_p 0.9 --max_new_tokens 1024",
            "description": "Benchmark with custom generation parameters"
        }
    ]
    
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1]) - 1
            if 0 <= idx < len(examples):
                example = examples[idx]
                print(f"Running example {idx + 1}: {example['name']}")
                run_command(example['cmd'], example['description'])
                return
            else:
                print(f"Invalid example number. Choose 1-{len(examples)}")
                sys.exit(1)
        except ValueError:
            print("Invalid example number. Please provide a number.")
            sys.exit(1)
    
    print("OLMo 2 Benchmarking Examples")
    print("=" * 40)
    print("Available examples:")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Command: {example['cmd']}")
        print(f"   Description: {example['description']}")
    
    print(f"\nUsage:")
    print(f"  python examples.py <number>    # Run specific example")
    print(f"  python examples.py            # Show this help")
    print(f"\nExample: python examples.py 1")
    
    # Show model comparison info
    print(f"\n" + "="*60)
    print("OLMo 2 Model Comparison:")
    print("="*60)
    
    models = [
        ("OLMo 2 1B", "allenai/OLMo-2-0425-1B", "4T tokens, 16 layers, 2048 hidden"),
        ("OLMo 2 7B", "allenai/OLMo-2-1124-7B", "4T tokens, 32 layers, 4096 hidden"),
        ("OLMo 2 13B", "allenai/OLMo-2-1124-13B", "5T tokens, 40 layers, 5120 hidden"),
        ("OLMo 2 32B", "allenai/OLMo-2-0325-32B", "6T tokens, 64 layers, 5120 hidden"),
    ]
    
    for name, model_id, specs in models:
        print(f"{name:12} | {model_id:30} | {specs}")
    
    print(f"\nModel variants (append to base name):")
    print(f"  -SFT      : Supervised fine-tuned")
    print(f"  -DPO      : Direct Preference Optimization") 
    print(f"  -Instruct : Final instruction-tuned (RLVR)")

if __name__ == "__main__":
    main()