#!/usr/bin/env python3
"""
Batch experiment runner for OLMo benchmarking
Runs all configured experiments systematically
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from experiments_config import get_experiments, ExperimentConfig, EXPERIMENT_GROUPS


class ExperimentRunner:
    def __init__(self, 
                 output_dir: str = "./experiment_results",
                 dry_run: bool = False,
                 continue_on_error: bool = True,
                 max_retries: int = 2):
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.max_retries = max_retries
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize experiment log
        self.experiment_log = []
        self.start_time = datetime.now()
        
    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment and return results"""
        experiment_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Running: {config.description}")
        print(f"Model: {config.model_name}")
        if config.revision:
            print(f"Revision: {config.revision}")
        print(f"Batch size: {config.batch_size}")
        print(f"{'='*60}")
        
        # Build command
        cmd = self._build_command(config)
        
        if self.dry_run:
            print(f"DRY RUN: Would execute: {' '.join(cmd)}")
            return {
                "config": config.__dict__,
                "status": "dry_run",
                "duration": 0,
                "command": ' '.join(cmd)
            }
        
        # Execute experiment with retries
        for attempt in range(self.max_retries + 1):
            try:
                print(f"Executing command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                duration = time.time() - experiment_start
                
                if result.returncode == 0:
                    # Success - parse results
                    print("✅ Experiment completed successfully")
                    return {
                        "config": config.__dict__,
                        "status": "success",
                        "duration": duration,
                        "command": ' '.join(cmd),
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "attempt": attempt + 1
                    }
                else:
                    print(f"❌ Experiment failed (attempt {attempt + 1}/{self.max_retries + 1})")
                    print(f"Return code: {result.returncode}")
                    print(f"STDERR: {result.stderr}")
                    
                    if attempt < self.max_retries:
                        print(f"Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        return {
                            "config": config.__dict__,
                            "status": "failed",
                            "duration": duration,
                            "command": ' '.join(cmd),
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "return_code": result.returncode,
                            "attempts": self.max_retries + 1
                        }
                        
            except subprocess.TimeoutExpired:
                duration = time.time() - experiment_start
                print(f"⏰ Experiment timed out (attempt {attempt + 1}/{self.max_retries + 1})")
                
                if attempt < self.max_retries:
                    print(f"Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    return {
                        "config": config.__dict__,
                        "status": "timeout",
                        "duration": duration,
                        "command": ' '.join(cmd),
                        "attempts": self.max_retries + 1
                    }
                    
            except Exception as e:
                duration = time.time() - experiment_start
                print(f"💥 Unexpected error: {str(e)}")
                return {
                    "config": config.__dict__,
                    "status": "error",
                    "duration": duration,
                    "command": ' '.join(cmd),
                    "error": str(e),
                    "attempt": attempt + 1
                }
        
    def _build_command(self, config: ExperimentConfig) -> List[str]:
        """Build the command to run the experiment"""
        cmd = ["./run_benchmark.sh", config.model_name]
        
        # Add revision if specified
        if config.revision:
            cmd.append(config.revision)
        else:
            cmd.append("''")  # Empty revision
        
        # Add additional arguments
        cmd.extend([
            "--batch_size", str(config.batch_size),
            "--max_questions", str(config.max_questions),
            "--output_dir", str(self.output_dir / "benchmark_results")
        ])
        
        return cmd
    
    def run_experiments(self, experiments: List[ExperimentConfig]) -> Dict[str, Any]:
        """Run all experiments and return summary"""
        print(f"🚀 Starting {len(experiments)} experiments...")
        print(f"Output directory: {self.output_dir}")
        print(f"Dry run: {self.dry_run}")
        print(f"Continue on error: {self.continue_on_error}")
        
        results = []
        successful = 0
        failed = 0
        skipped = 0
        
        for i, config in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Starting experiment...")
            
            try:
                result = self.run_single_experiment(config)
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                elif result["status"] == "dry_run":
                    skipped += 1
                else:
                    failed += 1
                    if not self.continue_on_error:
                        print(f"❌ Stopping due to failure (continue_on_error=False)")
                        break
                        
            except KeyboardInterrupt:
                print(f"\n⏹️  Interrupted by user")
                break
            except Exception as e:
                print(f"💥 Unexpected error in experiment runner: {str(e)}")
                failed += 1
                if not self.continue_on_error:
                    break
        
        # Create summary
        total_duration = time.time() - self.start_time.timestamp()
        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration": total_duration,
            "total_experiments": len(experiments),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "results": results
        }
        
        # Save summary
        summary_file = self.output_dir / f"experiment_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if skipped > 0:
            print(f"Skipped (dry run): {skipped}")
        print(f"Duration: {total_duration/3600:.2f} hours")
        print(f"Summary saved to: {summary_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run OLMo benchmark experiments")
    parser.add_argument("--group", type=str, default="all",
                       choices=list(EXPERIMENT_GROUPS.keys()),
                       help="Experiment group to run")
    parser.add_argument("--output_dir", type=str, default="./experiment_results",
                       help="Output directory for results")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show commands without executing")
    parser.add_argument("--stop_on_error", action="store_true",
                       help="Stop on first error (default: continue)")
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum retries per experiment")
    parser.add_argument("--list_groups", action="store_true",
                       help="List available experiment groups")
    
    args = parser.parse_args()
    
    if args.list_groups:
        print("Available experiment groups:")
        for group_name, experiments in EXPERIMENT_GROUPS.items():
            print(f"  {group_name}: {len(experiments)} experiments")
        return
    
    # Get experiments
    try:
        experiments = get_experiments(args.group)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Selected group: {args.group}")
    print(f"Number of experiments: {len(experiments)}")
    
    if not experiments:
        print("No experiments to run!")
        return 1
    
    # Create runner and execute
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
        max_retries=args.max_retries
    )
    
    summary = runner.run_experiments(experiments)
    
    # Return appropriate exit code
    if summary["failed"] > 0 and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())