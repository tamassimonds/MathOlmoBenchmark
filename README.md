# OLMo 2 Math Benchmark

A comprehensive benchmarking suite for evaluating OLMo 2 models on the MATH dataset. Optimized for dual B200 GPU setups with efficient batching and distributed inference.

## Features

- =€ **Multi-GPU Support**: Automatic distributed inference across multiple GPUs
- =Ê **Comprehensive Metrics**: Detailed scoring using mathematical equivalence checking
- = **Checkpoint Support**: Test different model checkpoints and revisions
- ¡ **Optimized Batching**: Efficient batch processing for B200 GPUs
- =È **Detailed Logging**: Complete results with timing and accuracy metrics
- <¯ **Flexible Configuration**: Customizable generation parameters

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MathOlmoBench

# Install dependencies
pip install -e .

# For 8-bit quantization support (optional)
pip install -e .[gpu]
```

### Basic Usage

```bash
# Benchmark OLMo 2 1B base model
./run_benchmark.sh allenai/OLMo-2-0425-1B

# Benchmark with specific checkpoint
./run_benchmark.sh allenai/OLMo-2-0425-1B stage1-step140000-tokens294B

# Benchmark instruct model with custom settings
./run_benchmark.sh allenai/OLMo-2-0425-1B-Instruct '' --batch_size 16 --max_questions 1000
```

### Run Examples

```bash
# See all available examples
python examples.py

# Run specific example
python examples.py 1
```

## Available Models

| Model | HuggingFace ID | Training Tokens | Parameters |
|-------|----------------|-----------------|------------|
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` | 4T | 1B |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` | 4T | 7B |
| OLMo 2 13B | `allenai/OLMo-2-1124-13B` | 5T | 13B |
| OLMo 2 32B | `allenai/OLMo-2-0325-32B` | 6T | 32B |

### Model Variants

Each base model has several variants:
- **Base**: Pre-trained model
- **SFT**: Supervised fine-tuned (`-SFT` suffix)
- **DPO**: Direct Preference Optimization (`-DPO` suffix)  
- **Instruct**: Final instruction-tuned (`-Instruct` suffix)

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | Required | HuggingFace model identifier |
| `--revision` | None | Specific checkpoint/revision |
| `--max_questions` | 500 | Number of questions to test |
| `--batch_size` | 8/16 | Batch size (auto-adjusted for multi-GPU) |
| `--max_new_tokens` | 512 | Maximum tokens to generate |
| `--temperature` | 0.0 | Sampling temperature (0.0 = greedy) |
| `--top_k` | 50 | Top-k sampling |
| `--top_p` | 0.95 | Top-p sampling |
| `--load_in_8bit` | False | Enable 8-bit quantization |
| `--torch_dtype` | float16 | Model precision |
| `--output_dir` | ./benchmark_results | Results directory |

## Output Format

The benchmark generates two types of files:

### Results File (`*_results_*.json`)
```json
[
  {
    "index": 0,
    "problem": "What is 2+2?",
    "ground_truth": "4",
    "response": "Let's solve this step by step...",
    "score": 1.0,
    "status": "OK",
    "batch_time": 0.15
  }
]
```

### Summary File (`*_summary_*.json`)
```json
{
  "model_name": "allenai/OLMo-2-0425-1B",
  "revision": null,
  "total_questions": 500,
  "correct_answers": 214,
  "accuracy": 0.428,
  "total_time": 245.6,
  "avg_time_per_question": 0.49,
  "config": {...}
}
```

## Performance Optimization

### Dual B200 Setup
The benchmark automatically optimizes for dual B200 GPUs:
- Batch size scales to 16 per GPU pair
- Distributed inference with `torchrun`
- Memory-efficient attention patterns
- Optimal tensor parallelism

### Memory Management
```bash
# For large models, use 8-bit quantization
./run_benchmark.sh allenai/OLMo-2-1124-7B '' --load_in_8bit

# Reduce batch size for 13B+ models
./run_benchmark.sh allenai/OLMo-2-1124-13B '' --batch_size 4
```

## Advanced Usage

### Custom Dataset
```python
from olmo2_benchmark import OLMo2Benchmark, BenchmarkConfig

config = BenchmarkConfig(
    model_name="allenai/OLMo-2-0425-1B",
    dataset_name="your-custom/math-dataset",
    max_questions=1000
)

benchmark = OLMo2Benchmark(config)
results = benchmark.run_benchmark()
```

### Programmatic Access
```python
import json
from olmo2_benchmark import OLMo2Benchmark, BenchmarkConfig

# Load existing results
with open("benchmark_results/summary.json") as f:
    summary = json.load(f)
    
print(f"Accuracy: {summary['accuracy']:.3f}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   ./run_benchmark.sh <model> '' --batch_size 2 --load_in_8bit
   ```

2. **Slow Performance**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Increase batch size if memory allows
   ./run_benchmark.sh <model> '' --batch_size 32
   ```

3. **Model Loading Issues**
   ```bash
   # Clear HuggingFace cache
   rm -rf ~/.cache/huggingface/
   
   # Update transformers
   pip install --upgrade transformers
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Citation

If you use this benchmarking suite, please cite:

```bibtex
@misc{olmo2benchmark2024,
  title={OLMo 2 Math Benchmark Suite},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/MathOlmoBench}
}
```# MathOlmoBenchmark
