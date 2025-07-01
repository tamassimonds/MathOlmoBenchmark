#!/usr/bin/env python3
"""
Results analysis and aggregation script for OLMo benchmarking
Processes experiment results and generates comprehensive reports
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


class ResultsAnalyzer:
    """Analyze and visualize OLMo benchmarking results"""
    
    def __init__(self, results_dir: str = "./experiment_results"):
        self.results_dir = Path(results_dir)
        self.benchmark_results_dir = self.results_dir / "benchmark_results"
        
    def load_experiment_summary(self, summary_file: Optional[str] = None) -> Dict[str, Any]:
        """Load experiment summary file"""
        if summary_file:
            summary_path = Path(summary_file)
        else:
            # Find the most recent summary file
            summary_files = list(self.results_dir.glob("experiment_summary_*.json"))
            if not summary_files:
                raise FileNotFoundError("No experiment summary files found")
            summary_path = max(summary_files, key=lambda p: p.stat().st_mtime)
            
        print(f"Loading experiment summary: {summary_path}")
        with open(summary_path, 'r') as f:
            return json.load(f)
            
    def load_benchmark_results(self) -> List[Dict[str, Any]]:
        """Load all benchmark result files"""
        results = []
        
        if not self.benchmark_results_dir.exists():
            print(f"Warning: Benchmark results directory not found: {self.benchmark_results_dir}")
            return results
            
        # Load summary files
        summary_files = list(self.benchmark_results_dir.glob("*_summary_*.json"))
        
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    results.append(summary)
            except Exception as e:
                print(f"Warning: Failed to load {summary_file}: {e}")
                
        print(f"Loaded {len(results)} benchmark result files")
        return results
        
    def create_results_dataframe(self, benchmark_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert benchmark results to pandas DataFrame"""
        data = []
        
        for result in benchmark_results:
            # Extract model info
            model_name = result.get('model_name', '')
            revision = result.get('revision', '')
            
            # Parse model info
            model_size = "Unknown"
            model_type = "Base"
            
            if "1B" in model_name:
                model_size = "1B"
            elif "7B" in model_name:
                model_size = "7B" 
            elif "13B" in model_name:
                model_size = "13B"
            elif "32B" in model_name:
                model_size = "32B"
                
            if "-SFT" in model_name:
                model_type = "SFT"
            elif "-DPO" in model_name:
                model_type = "DPO"
            elif "-Instruct" in model_name:
                model_type = "Instruct"
                
            # Extract training step from revision
            training_step = 0
            tokens_trained = "0B"
            if revision and "step" in revision:
                try:
                    step_part = revision.split("step")[1].split("-")[0]
                    training_step = int(step_part)
                    
                    if "tokens" in revision:
                        tokens_part = revision.split("tokens")[1].split("B")[0]
                        tokens_trained = tokens_part + "B"
                except:
                    pass
                    
            data.append({
                'model_name': model_name,
                'model_size': model_size,
                'model_type': model_type,
                'revision': revision,
                'training_step': training_step,
                'tokens_trained': tokens_trained,
                'accuracy': result.get('accuracy', 0),
                'total_questions': result.get('total_questions', 0),
                'correct_answers': result.get('correct_answers', 0),
                'total_time': result.get('total_time', 0),
                'avg_time_per_question': result.get('avg_time_per_question', 0),
            })
            
        return pd.DataFrame(data)
        
    def generate_report(self, output_file: str = "benchmark_report.html") -> str:
        """Generate comprehensive HTML report"""
        
        # Load data
        try:
            experiment_summary = self.load_experiment_summary()
        except FileNotFoundError:
            experiment_summary = {"results": []}
            
        benchmark_results = self.load_benchmark_results()
        df = self.create_results_dataframe(benchmark_results)
        
        # Generate HTML report
        html_content = self._generate_html_report(experiment_summary, df)
        
        # Save report
        report_path = self.results_dir / output_file
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return str(report_path)
        
    def _generate_html_report(self, experiment_summary: Dict, df: pd.DataFrame) -> str:
        """Generate HTML content for the report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OLMo Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 30px 0; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 20px; background: #ecf0f1; border-radius: 8px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .best-score {{ background: #d5f4e6; font-weight: bold; }}
        .chart-placeholder {{ background: #f8f9fa; padding: 40px; text-align: center; 
                            border: 2px dashed #dee2e6; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧮 OLMo Math Benchmark Report</h1>
        <p>Generated on {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>📊 Experiment Overview</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Experiments</div>
            </div>
            <div class="metric">
                <div class="metric-value">{df['accuracy'].max():.3f}</div>
                <div class="metric-label">Best Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(df['model_size'].unique())}</div>
                <div class="metric-label">Model Sizes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{df['total_questions'].sum():,}</div>
                <div class="metric-label">Total Questions</div>
            </div>
        </div>
    </div>
"""
        
        if not df.empty:
            # Best performing models
            html += self._generate_best_models_section(df)
            
            # Results by model size
            html += self._generate_model_size_section(df)
            
            # Training progression analysis
            html += self._generate_training_progression_section(df)
            
            # Detailed results table
            html += self._generate_detailed_table(df)
            
        else:
            html += """
    <div class="section">
        <h2>⚠️ No Results Found</h2>
        <p>No benchmark results were found. Please run some experiments first.</p>
    </div>
"""
        
        html += """
    <div class="section">
        <h2>📝 Notes</h2>
        <ul>
            <li>Accuracy is measured on the MATH dataset using mathematical equivalence checking</li>
            <li>Training steps and token counts are extracted from model revisions</li>
            <li>Times are measured in seconds per question</li>
            <li>Best scores are highlighted in green</li>
        </ul>
    </div>
    
</body>
</html>
"""
        
        return html
        
    def _generate_best_models_section(self, df: pd.DataFrame) -> str:
        """Generate best performing models section"""
        best_overall = df.loc[df['accuracy'].idxmax()]
        best_by_size = df.groupby('model_size')['accuracy'].idxmax()
        
        html = """
    <div class="section">
        <h2>🏆 Best Performing Models</h2>
        <h3>Overall Best</h3>
        <p><strong>{}</strong> ({}): {:.3f} accuracy</p>
        
        <h3>Best by Model Size</h3>
        <ul>
""".format(best_overall['model_name'], best_overall['revision'] or 'Base', best_overall['accuracy'])
        
        for size_idx in best_by_size:
            model = df.loc[size_idx]
            html += f"            <li><strong>{model['model_size']}</strong>: {model['model_name']} ({model['revision'] or 'Base'}) - {model['accuracy']:.3f}</li>\n"
            
        html += """        </ul>
    </div>
"""
        return html
        
    def _generate_model_size_section(self, df: pd.DataFrame) -> str:
        """Generate model size comparison section"""
        size_stats = df.groupby('model_size').agg({
            'accuracy': ['mean', 'max', 'std'],
            'avg_time_per_question': 'mean'
        }).round(3)
        
        html = """
    <div class="section">
        <h2>📏 Performance by Model Size</h2>
        <table>
            <tr>
                <th>Model Size</th>
                <th>Mean Accuracy</th>
                <th>Max Accuracy</th>
                <th>Std Dev</th>
                <th>Avg Time/Question (s)</th>
            </tr>
"""
        
        for size in size_stats.index:
            stats = size_stats.loc[size]
            html += f"""            <tr>
                <td>{size}</td>
                <td>{stats[('accuracy', 'mean')]:.3f}</td>
                <td>{stats[('accuracy', 'max')]:.3f}</td>
                <td>{stats[('accuracy', 'std')]:.3f}</td>
                <td>{stats[('avg_time_per_question', 'mean')]:.2f}</td>
            </tr>
"""
        
        html += """        </table>
    </div>
"""
        return html
        
    def _generate_training_progression_section(self, df: pd.DataFrame) -> str:
        """Generate training progression analysis"""
        # Filter for base models with training steps
        base_models = df[(df['model_type'] == 'Base') & (df['training_step'] > 0)]
        
        if base_models.empty:
            return ""
            
        html = """
    <div class="section">
        <h2>📈 Training Progression Analysis</h2>
        <p>Accuracy improvement during training for base models:</p>
        <div class="chart-placeholder">
            Training progression charts would be displayed here<br>
            (Requires matplotlib integration for web display)
        </div>
"""
        
        # Show progression for each model size
        for size in base_models['model_size'].unique():
            size_data = base_models[base_models['model_size'] == size].sort_values('training_step')
            if len(size_data) > 1:
                initial_acc = size_data.iloc[0]['accuracy']
                final_acc = size_data.iloc[-1]['accuracy']
                improvement = final_acc - initial_acc
                html += f"        <p><strong>{size}</strong>: {improvement:.3f} accuracy improvement from step {size_data.iloc[0]['training_step']:,} to {size_data.iloc[-1]['training_step']:,}</p>\n"
                
        html += """    </div>
"""
        return html
        
    def _generate_detailed_table(self, df: pd.DataFrame) -> str:
        """Generate detailed results table"""
        # Sort by accuracy descending
        df_sorted = df.sort_values('accuracy', ascending=False)
        best_accuracy = df_sorted.iloc[0]['accuracy']
        
        html = """
    <div class="section">
        <h2>📋 Detailed Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Size</th>
                <th>Type</th>
                <th>Training Step</th>
                <th>Accuracy</th>
                <th>Correct/Total</th>
                <th>Avg Time (s)</th>
            </tr>
"""
        
        for _, row in df_sorted.iterrows():
            is_best = row['accuracy'] == best_accuracy
            row_class = ' class="best-score"' if is_best else ''
            
            model_display = row['model_name'].replace('allenai/', '')
            revision_display = f"Step {row['training_step']:,}" if row['training_step'] > 0 else "Base"
            
            html += f"""            <tr{row_class}>
                <td>{model_display}</td>
                <td>{row['model_size']}</td>
                <td>{row['model_type']}</td>
                <td>{revision_display}</td>
                <td>{row['accuracy']:.3f}</td>
                <td>{row['correct_answers']}/{row['total_questions']}</td>
                <td>{row['avg_time_per_question']:.2f}</td>
            </tr>
"""
        
        html += """        </table>
    </div>
"""
        return html
        
    def export_csv(self, output_file: str = "benchmark_results.csv") -> str:
        """Export results to CSV"""
        benchmark_results = self.load_benchmark_results()
        df = self.create_results_dataframe(benchmark_results)
        
        csv_path = self.results_dir / output_file
        df.to_csv(csv_path, index=False)
        
        print(f"Results exported to: {csv_path}")
        return str(csv_path)
        
    def create_plots(self, output_dir: str = "plots"):
        """Create visualization plots"""
        benchmark_results = self.load_benchmark_results()
        df = self.create_results_dataframe(benchmark_results)
        
        if df.empty:
            print("No data to plot")
            return
            
        plots_dir = self.results_dir / output_dir
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Accuracy by model size
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='model_size', y='accuracy')
        plt.title('Accuracy by Model Size')
        plt.ylabel('Accuracy')
        plt.xlabel('Model Size')
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_by_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training progression for base models
        base_models = df[(df['model_type'] == 'Base') & (df['training_step'] > 0)]
        if not base_models.empty:
            plt.figure(figsize=(12, 8))
            for size in base_models['model_size'].unique():
                size_data = base_models[base_models['model_size'] == size].sort_values('training_step')
                plt.plot(size_data['training_step'], size_data['accuracy'], 
                        marker='o', label=f'{size} Model', linewidth=2)
            
            plt.title('Training Progression: Accuracy vs Training Steps')
            plt.xlabel('Training Steps')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'training_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Model type comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='model_type', y='accuracy', hue='model_size')
        plt.title('Accuracy by Model Type and Size')
        plt.ylabel('Accuracy')
        plt.xlabel('Model Type')
        plt.legend(title='Model Size')
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_by_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze OLMo benchmark results")
    parser.add_argument("--results_dir", type=str, default="./experiment_results",
                       help="Directory containing experiment results")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("--csv", action="store_true",
                       help="Export results to CSV")
    parser.add_argument("--plots", action="store_true",
                       help="Create visualization plots")
    parser.add_argument("--all", action="store_true",
                       help="Generate all outputs (report, CSV, plots)")
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if args.all or args.report:
        report_path = analyzer.generate_report()
        print(f"HTML report generated: {report_path}")
        
    if args.all or args.csv:
        csv_path = analyzer.export_csv()
        print(f"CSV exported: {csv_path}")
        
    if args.all or args.plots:
        analyzer.create_plots()
        print("Plots created")
        
    if not any([args.report, args.csv, args.plots, args.all]):
        print("No output specified. Use --report, --csv, --plots, or --all")


if __name__ == "__main__":
    main()