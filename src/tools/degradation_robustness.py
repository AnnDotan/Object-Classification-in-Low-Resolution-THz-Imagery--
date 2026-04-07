"""
Degradation Robustness Visualization

Compares model performance across different degradation types:
- DOWNSAMPLING: Low resolution only
- BLUR: Gaussian blur only
- NOISE: Gaussian noise only

Shows how each model handles specific degradation types.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


class DegradationRobustnessAnalysis:
    """Analyzes and visualizes robustness to specific degradation types."""

    DEGRADATION_TYPES = {
        'downsampling': 'Downsampling (Low-Res)',
        'blur': 'Gaussian Blur',
        'noise': 'Gaussian Noise'
    }

    def __init__(self, runs_dir: Path = Path("runs")):
        """Initialize analyzer."""
        self.runs_dir = Path(runs_dir)

    def find_runs_by_degradation(self, model_name: str, degradation_type: str) -> Optional[Path]:
        """
        Find the run directory for a specific model and degradation type.

        Args:
            model_name: 'resnet50', 'densenet121', or 'transnext_micro'
            degradation_type: 'downsampling', 'blur', or 'noise'

        Returns:
            Path to run directory, or None if not found
        """
        for run_dir in self.runs_dir.rglob("*"):
            if not run_dir.is_dir():
                continue

            metrics_file = run_dir / "metrics.csv"
            run_config = run_dir / "run_config.txt"

            if not (metrics_file.exists() and run_config.exists()):
                continue

            # Check if this run matches the model and degradation type
            config_text = run_config.read_text(encoding='utf-8', errors='ignore')

            # Check model name
            if model_name.lower() not in config_text.lower():
                continue

            # Check degradation type
            if degradation_type.lower() not in run_dir.name.lower():
                continue

            return run_dir

        return None

    def load_metrics(self, run_dir: Path) -> Optional[pd.DataFrame]:
        """Load metrics from a run."""
        metrics_file = run_dir / "metrics.csv"
        if not metrics_file.exists():
            return None

        try:
            return pd.read_csv(metrics_file)
        except:
            return None

    def plot_model_degradation_comparison(self, model_name: str, figsize: tuple = (15, 5),
                                         dpi: int = 100) -> Optional[Path]:
        """
        Create comparison plot showing how one model performs across degradation types.

        Args:
            model_name: 'resnet50', 'densenet121', or 'transnext_micro'
            figsize: Figure size (width, height)
            dpi: DPI for saved figure

        Returns:
            Path to saved figure
        """
        print(f"\n[INFO] Analyzing {model_name} robustness to degradation types...")

        # Find runs for each degradation type
        runs = {}
        metrics_data = {}

        for deg_type in self.DEGRADATION_TYPES.keys():
            run_dir = self.find_runs_by_degradation(model_name, deg_type)
            if run_dir:
                metrics = self.load_metrics(run_dir)
                if metrics is not None and len(metrics) > 0:
                    runs[deg_type] = run_dir
                    metrics_data[deg_type] = metrics
                    print(f"  [OK] Found {deg_type}: {run_dir.name}")
                else:
                    print(f"  [WARN] {deg_type}: No metrics found")
            else:
                print(f"  [WARN] {deg_type}: Run not found")

        if not metrics_data:
            print(f"[ERROR] No runs found for {model_name}")
            return None

        # Create figure with subplots (one per degradation type)
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        if not isinstance(axes, list):
            axes = [axes]

        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

        max_epoch = 0

        for idx, (deg_type, deg_label) in enumerate(self.DEGRADATION_TYPES.items()):
            ax = axes[idx]

            if deg_type in metrics_data:
                metrics = metrics_data[deg_type]
                epochs = metrics['epoch'].values if 'epoch' in metrics.columns else range(len(metrics))
                max_epoch = max(max_epoch, max(epochs) if len(epochs) > 0 else 0)

                if 'val_acc' in metrics.columns:
                    # Plot validation accuracy
                    ax.plot(epochs, metrics['val_acc'] * 100, marker='o', color=colors[idx],
                           linewidth=2.5, markersize=6, label='Val Accuracy', alpha=0.8)

                    # Mark best point
                    best_idx = metrics['val_acc'].idxmax()
                    best_acc = metrics['val_acc'].iloc[best_idx] * 100
                    best_epoch = epochs[best_idx]
                    ax.plot(best_epoch, best_acc, marker='*', color=colors[idx],
                           markersize=25, label=f'Best: {best_acc:.2f}%')

                    # Also plot training accuracy
                    if 'train_acc' in metrics.columns:
                        ax.plot(epochs, metrics['train_acc'] * 100, marker='s',
                               color=colors[idx], linewidth=2, markersize=5,
                               linestyle='--', alpha=0.6, label='Train Accuracy')

            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax.set_title(deg_label, fontsize=12, fontweight='bold')
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
            ax.set_xlim([0, max_epoch + 1])

        # Overall title
        fig.suptitle(f'{model_name.upper()} - Robustness to Degradation Types',
                    fontsize=14, fontweight='bold')

        # Save
        output_path = Path("artifacts/figures") / f"degradation_robustness_{model_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved: {output_path}\n")
        return output_path

    def plot_all_models_degradation(self, models: List[str] = None,
                                   figsize: tuple = (16, 12), dpi: int = 100):
        """Generate degradation robustness plots for all models."""
        if models is None:
            models = ['resnet50', 'densenet121', 'transnext_micro']

        generated = []

        for model in models:
            path = self.plot_model_degradation_comparison(model, figsize=figsize, dpi=dpi)
            if path:
                generated.append(path)

        print(f"\n[OK] Generated {len(generated)} degradation robustness plots")
        return generated

    def plot_degradation_summary_table(self, models: List[str] = None,
                                     figsize: tuple = (14, 8), dpi: int = 100) -> Optional[Path]:
        """Create a summary table comparing all models and degradation types."""
        if models is None:
            models = ['resnet50', 'densenet121', 'transnext_micro']

        print("[INFO] Creating degradation summary table...")

        # Collect data
        summary_data = []

        for model in models:
            for deg_type in self.DEGRADATION_TYPES.keys():
                run_dir = self.find_runs_by_degradation(model, deg_type)
                if run_dir:
                    metrics = self.load_metrics(run_dir)
                    if metrics is not None and len(metrics) > 0:
                        best_acc = metrics['val_acc'].max() * 100
                        summary_data.append({
                            'Model': model.replace('_', ' ').upper(),
                            'Degradation': self.DEGRADATION_TYPES[deg_type],
                            'Best Val Acc': f"{best_acc:.2f}%",
                            'Epochs': len(metrics)
                        })

        if not summary_data:
            print("[WARN] No data to create summary")
            return None

        # Create figure with table
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis('off')

        # Create table text
        summary_text = "DEGRADATION ROBUSTNESS SUMMARY\n"
        summary_text += "=" * 70 + "\n\n"

        # Organize by model
        models_set = set(row['Model'] for row in summary_data)

        for model in sorted(models_set):
            summary_text += f"{model}\n"
            summary_text += "-" * 70 + "\n"

            for row in summary_data:
                if row['Model'] == model:
                    summary_text += f"  {row['Degradation']:<25} {row['Best Val Acc']:>10}  ({row['Epochs']} epochs)\n"

            summary_text += "\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

        output_path = Path("artifacts/figures/degradation_robustness_summary.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"[OK] Summary saved: {output_path}\n")
        return output_path


def plot_degradation_analysis(models: List[str] = None) -> list:
    """Convenience function to generate all degradation robustness plots."""
    analyzer = DegradationRobustnessAnalysis()

    if models is None:
        models = ['resnet50', 'densenet121', 'transnext_micro']

    generated = analyzer.plot_all_models_degradation(models)
    analyzer.plot_degradation_summary_table(models)

    return generated


if __name__ == "__main__":
    plot_degradation_analysis()
