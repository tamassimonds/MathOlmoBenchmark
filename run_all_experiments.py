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
                 max_retries: int = 2,
                 debug: bool = False):
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.max_retries = max_retries
        self.debug = debug
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize experiment log
        self.experiment_log = []
        self.start_time = datetime.now()
        self.log_file = self.output_dir / f"experiment_log_{int(self.start_time.timestamp())}.txt"
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"OLMo Experiment Log - Started at {self.start_time.isoformat()}\n")
            f.write("=" * 80 + "\n\n")
        
    def log_and_print(self, message: str, also_print: bool = True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
            
        if also_print:
            print(f"[{timestamp}] {message}")
    
    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment and return results"""
        experiment_start = time.time()
        
        # Clear previous prints and show current experiment
        os.system('clear' if os.name == 'posix' else 'cls')
        
        header = f"Running: {config.description}"
        self.log_and_print("=" * 60)
        self.log_and_print(header)
        self.log_and_print(f"Model: {config.model_name}")
        if config.revision:
            self.log_and_print(f"Revision: {config.revision}")
        self.log_and_print(f"Batch size: {config.batch_size}")
        self.log_and_print("=" * 60)
        
        # Build command
        cmd = self._build_command(config)
        
        if self.dry_run:
            self.log_and_print(f"DRY RUN: Would execute: {' '.join(cmd)}")
            return {
                "config": config.__dict__,
                "status": "dry_run",
                "duration": 0,
                "command": ' '.join(cmd)
            }
        
        # Execute experiment (no retries)
        try:
            self.log_and_print(f"Executing command: {' '.join(cmd)}")
            print(f"RUNNING: {' '.join(cmd)}")  # Direct print to terminal
            result = subprocess.run(cmd, timeout=3600)
            
            duration = time.time() - experiment_start
            success_msg = f"✅ Experiment completed successfully in {duration:.1f}s"
            self.log_and_print(success_msg)
            
            return {
                "config": config.__dict__,
                "status": "success", 
                "duration": duration,
                "command": ' '.join(cmd)
            }
            
        except Exception as e:
            duration = time.time() - experiment_start
            error_msg = f"💥 Experiment failed: {str(e)}"
            print(f"ERROR: {error_msg}")  # Direct print to terminal
            self.log_and_print(error_msg)
            
            return {
                "config": config.__dict__,
                "status": "failed",
                "duration": duration,
                "command": ' '.join(cmd),
                "error": str(e)
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
    
    def _save_individual_result(self, config: ExperimentConfig, result: Dict[str, Any]):
        """Save individual experiment result immediately"""
        timestamp = int(time.time())
        result_file = self.output_dir / f"result_{config.model_name.replace('/', '_')}_{timestamp}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            self.log_and_print(f"💾 Result saved to: {result_file.name}", False)
        except Exception as e:
            self.log_and_print(f"⚠️  Failed to save result: {e}", False)
    
    def run_experiments(self, experiments: List[ExperimentConfig]) -> Dict[str, Any]:
        """Run all experiments and return summary"""
        start_msg = f"🚀 Starting {len(experiments)} experiments..."
        self.log_and_print(start_msg)
        self.log_and_print(f"Output directory: {self.output_dir}")
        self.log_and_print(f"Log file: {self.log_file}")
        self.log_and_print(f"Dry run: {self.dry_run}")
        self.log_and_print(f"Continue on error: {self.continue_on_error}")
        
        results = []
        successful = 0
        failed = 0
        skipped = 0
        
        for i, config in enumerate(experiments, 1):
            progress_msg = f"\n[{i}/{len(experiments)}] Starting experiment..."
            self.log_and_print(progress_msg)
            
            try:
                result = self.run_single_experiment(config)
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                    self.log_and_print(f"✅ Progress: {successful} success, {failed} failed, {len(experiments) - i} remaining")
                elif result["status"] == "dry_run":
                    skipped += 1
                else:
                    failed += 1
                    self.log_and_print(f"❌ Progress: {successful} success, {failed} failed, {len(experiments) - i} remaining")
                    if not self.continue_on_error:
                        self.log_and_print(f"❌ Stopping due to failure (continue_on_error=False)")
                        break
                        
            except KeyboardInterrupt:
                self.log_and_print(f"\n⏹️  Interrupted by user")
                break
            except Exception as e:
                self.log_and_print(f"💥 Unexpected error in experiment runner: {str(e)}")
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
        
        # Final summary
        self.log_and_print("\n" + "="*60)
        self.log_and_print("📊 EXPERIMENT SUMMARY")
        self.log_and_print("="*60)
        self.log_and_print(f"Total experiments: {len(experiments)}")
        self.log_and_print(f"Successful: {successful}")
        self.log_and_print(f"Failed: {failed}")
        if skipped > 0:
            self.log_and_print(f"Skipped (dry run): {skipped}")
        self.log_and_print(f"Duration: {total_duration/3600:.2f} hours")
        self.log_and_print(f"Summary saved to: {summary_file}")
        self.log_and_print(f"Full log saved to: {self.log_file}")
        
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
    parser.add_argument("--debug", action="store_true",
                       help="Show full error output for debugging")
    
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
        max_retries=args.max_retries,
        debug=args.debug
    )
    
    summary = runner.run_experiments(experiments)
    
    # Return appropriate exit code
    if summary["failed"] > 0 and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())