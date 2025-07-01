# Quick Start Guide: OLMo Benchmarking

This guide helps you quickly run all the OLMo experiments you specified.

## 🚀 Quick Commands

### Run All Experiments
```bash
# Run all experiments (will take many hours)
./run_all_experiments.py --group all

# Run just 1B model experiments
./run_all_experiments.py --group olmo_1b

# Run just 7B model experiments  
./run_all_experiments.py --group olmo_7b

# Run just 13B model experiments
./run_all_experiments.py --group olmo_13b
```

### Run Specific Model Types
```bash
# Run only base models (no fine-tuning)
./run_all_experiments.py --group base_models

# Run only SFT models
./run_all_experiments.py --group sft_models

# Run only instruct models
./run_all_experiments.py --group instruct_models
```

### Dry Run (Preview Commands)
```bash
# See what experiments would run without executing
./run_all_experiments.py --group all --dry_run
```

### Analyze Results
```bash
# Generate comprehensive HTML report
./analyze_results.py --report

# Export results to CSV
./analyze_results.py --csv

# Create visualization plots
./analyze_results.py --plots

# Do everything
./analyze_results.py --all
```

## 📊 Your Experiment Plan

Based on your specifications, here are the experiments that will run:

### OLMo 1B (12 experiments)
- **Base Model Checkpoints**: Steps 0, 10k, 100k, 500k, 1M, 1.5M, 1.9M
- **Stage 2**: Step 23,852
- **Fine-tuned**: SFT, DPO, Instruct

### OLMo 7B (9 experiments)  
- **Base Model Checkpoints**: Steps 0, 10k, 100k, 500k, 750k, 928k
- **Fine-tuned**: SFT, DPO, Instruct

### OLMo 13B (9 experiments)
- **Base Model Checkpoints**: Steps 0, 10k, 100k, 300k, 450k, 596k  
- **Fine-tuned**: SFT, DPO, Instruct

**Total: 30 experiments**

## ⚙️ Configuration

The experiments are configured with:
- **Batch sizes**: 8 (1B), 4 (7B), 2 (13B) - optimized for memory
- **Questions**: 500 per experiment (15,000 total)
- **Auto-retry**: Failed experiments retry up to 2 times
- **Timeout**: 1 hour per experiment

## 📁 Output Structure

```
experiment_results/
├── experiment_summary_TIMESTAMP.json    # Overall experiment log
├── benchmark_results/                   # Individual model results
│   ├── allenai_OLMo-2-0425-1B_results_TIMESTAMP.json
│   ├── allenai_OLMo-2-0425-1B_summary_TIMESTAMP.json
│   └── ...
├── benchmark_report.html               # Generated report
├── benchmark_results.csv               # Exported data
└── plots/                             # Visualization charts
    ├── accuracy_by_size.png
    ├── training_progression.png
    └── accuracy_by_type.png
```

## 🔧 Advanced Options

### Resume Failed Experiments
```bash
# Continue from where you left off
./run_all_experiments.py --group all --continue_on_error
```

### Custom Settings
```bash
# Reduce questions for faster testing
./run_all_experiments.py --group olmo_1b --max_questions 100

# Use smaller batch sizes for memory constraints
./run_all_experiments.py --group olmo_7b --batch_size 2
```

### Monitor Progress
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor experiment directory
ls -la experiment_results/benchmark_results/
```

## ⚡ Estimated Runtime

- **1B models**: ~2-3 hours per experiment
- **7B models**: ~4-6 hours per experiment  
- **13B models**: ~6-8 hours per experiment

**Total estimated time: 4-6 days** (depending on hardware)

## 🎯 Recommended Workflow

1. **Start with a test run**:
   ```bash
   ./run_all_experiments.py --group olmo_1b --dry_run
   ```

2. **Run 1B experiments first**:
   ```bash
   ./run_all_experiments.py --group olmo_1b
   ```

3. **Analyze initial results**:
   ```bash
   ./analyze_results.py --report
   ```

4. **Run larger models**:
   ```bash
   ./run_all_experiments.py --group olmo_7b
   ./run_all_experiments.py --group olmo_13b
   ```

5. **Generate final report**:
   ```bash
   ./analyze_results.py --all
   ```

## 🆘 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
./run_all_experiments.py --group olmo_7b --batch_size 1

# Enable 8-bit quantization
./run_benchmark.sh allenai/OLMo-2-1124-7B '' --load_in_8bit
```

### Slow Downloads
The first run will download models (~4GB each). Subsequent runs use cached models.

### Check Available Models
```bash
python experiments_config.py  # Shows all configured experiments
```

This setup handles all your specified experiments automatically. Just run the commands and the system will execute everything systematically!