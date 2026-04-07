"""
Real-time learning curve visualization for individual experiment runs.

This module generates publication-quality learning curves after each training run,
enabling quick visual analysis for research decisions.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


class RunVisualizer:
    """Generates learning curves and metrics visualizations for a single experiment run."""

    def __init__(self, run_dir: Path, dpi: int = 100):
        """
        Initialize visualizer for a run directory.

        Args:
            run_dir: Path to experiment run directory (contains metrics.csv, log.txt)
            dpi: DPI for saved figures (default 100, use 150+ for publications)
        """
        self.run_dir = Path(run_dir)
        self.metrics_file = self.run_dir / "metrics.csv"
        self.log_file = self.run_dir / "log.txt"
        self.dpi = dpi

        # Output directory for visualizations (same as run_dir)
        self.output_dir = self.run_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_metrics(self) -> Optional[pd.DataFrame]:
        """Load metrics from CSV file."""
        if not self.metrics_file.exists():
            print(f"[WARN] Metrics file not found: {self.metrics_file}")
            return None

        try:
            df = pd.read_csv(self.metrics_file)
            if len(df) == 0:
                print(f"[WARN] Metrics file is empty: {self.metrics_file}")
                return None
            return df
        except Exception as e:
            print(f"[ERROR] Error loading metrics: {e}")
            return None

    def load_config(self) -> dict:
        """Load run configuration from log file."""
        config = {}
        if self.log_file.exists():
            try:
                content = self.log_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                for line in lines[:15]:  # First ~15 lines have config info
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        config[key.strip()] = value.strip()
            except:
                pass
        return config

    def plot_learning_curves(self) -> Optional[Path]:
        """
        Generate comprehensive learning curves plot.

        Shows:
        - Training loss vs epoch
        - Validation loss vs epoch
        - Training accuracy vs epoch
        - Validation accuracy vs epoch

        Returns:
            Path to saved figure, or None if failed
        """
        df = self.load_metrics()
        if df is None or len(df) < 2:
            return None

        config = self.load_config()

        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Extract data
        epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))

        # Color scheme
        color_train = '#2E86AB'  # Blue
        color_val = '#A23B72'    # Purple

        # ============= PLOT 1: Training Loss =============
        ax1 = fig.add_subplot(gs[0, 0])
        if 'train_loss' in df.columns:
            ax1.plot(epochs, df['train_loss'], marker='o', color=color_train,
                    linewidth=2, markersize=4, label='Training')
            ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='best', fontsize=10)

        # ============= PLOT 2: Validation Loss =============
        ax2 = fig.add_subplot(gs[0, 1])
        if 'val_loss' in df.columns:
            ax2.plot(epochs, df['val_loss'], marker='s', color=color_val,
                    linewidth=2, markersize=4, label='Validation')
            ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=10)

        # ============= PLOT 3: Training Accuracy =============
        ax3 = fig.add_subplot(gs[1, 0])
        if 'train_acc' in df.columns:
            ax3.plot(epochs, df['train_acc'] * 100, marker='o', color=color_train,
                    linewidth=2, markersize=4, label='Training')
            ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax3.set_title('Training Accuracy', fontsize=12, fontweight='bold')
            ax3.set_ylim([0, 105])
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.legend(loc='best', fontsize=10)

        # ============= PLOT 4: Validation Accuracy =============
        ax4 = fig.add_subplot(gs[1, 1])
        if 'val_acc' in df.columns:
            ax4.plot(epochs, df['val_acc'] * 100, marker='s', color=color_val,
                    linewidth=2, markersize=4, label='Validation')

            # Highlight best epoch
            best_idx = df['val_acc'].idxmax()
            best_epoch = epochs[best_idx]
            best_acc = df['val_acc'].iloc[best_idx] * 100
            ax4.plot(best_epoch, best_acc, marker='*', color='#F18F01',
                    markersize=20, label=f'Best: {best_acc:.2f}% (epoch {best_epoch})')

            ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax4.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 105])
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.legend(loc='best', fontsize=10)

        # Add overall title with run info
        model = config.get('Model', 'Unknown')
        low_res = config.get('Data', 'Unknown')
        fig.suptitle(f'Learning Curves: {model} | {low_res}',
                    fontsize=14, fontweight='bold', y=0.995)

        # Save figure
        output_path = self.output_dir / 'learning_curves.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[OK] Learning curves saved: {output_path}")
        return output_path

    def plot_loss_comparison(self) -> Optional[Path]:
        """
        Generate side-by-side train vs validation loss comparison.

        Helps identify overfitting quickly.
        """
        df = self.load_metrics()
        if df is None or len(df) < 2:
            return None

        config = self.load_config()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)

        epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))

        # Loss comparison
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            ax1.plot(epochs, df['train_loss'], marker='o', label='Training Loss',
                    linewidth=2.5, markersize=5, color='#2E86AB')
            ax1.plot(epochs, df['val_loss'], marker='s', label='Validation Loss',
                    linewidth=2.5, markersize=5, color='#A23B72')
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax1.set_title('Loss Curves (Overfitting Indicator)', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='--')

        # Accuracy comparison
        if 'train_acc' in df.columns and 'val_acc' in df.columns:
            ax2.plot(epochs, df['train_acc'] * 100, marker='o', label='Training Accuracy',
                    linewidth=2.5, markersize=5, color='#2E86AB')
            ax2.plot(epochs, df['val_acc'] * 100, marker='s', label='Validation Accuracy',
                    linewidth=2.5, markersize=5, color='#A23B72')
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
            ax2.set_ylim([0, 105])
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle='--')

        fig.suptitle(f'Overfitting Analysis: {config.get("Model", "Unknown")}',
                    fontsize=14, fontweight='bold')

        output_path = self.output_dir / 'overfitting_analysis.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[OK] Overfitting analysis saved: {output_path}")
        return output_path

    def plot_metrics_summary(self) -> Optional[Path]:
        """
        Generate a summary statistics visualization.

        Shows: best/worst/final metrics at a glance.
        """
        df = self.load_metrics()
        if df is None or len(df) < 1:
            return None

        config = self.load_config()

        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        ax.axis('off')

        # Prepare summary statistics
        summary_text = f"""
EXPERIMENT SUMMARY
{'='*60}

Model Configuration:
  • Model: {config.get('Model', 'N/A')}
  • Pretrained: {config.get('Pretrained', 'N/A')}
  • Device: {config.get('Device', 'N/A')}

Data Configuration:
  • {config.get('Data', 'N/A')}

Training Configuration:
  • Epochs: {len(df)}
  • Batch Size: {config.get('Batch Size', 'N/A')}
  • Learning Rate: {config.get('LR', 'N/A')}

Results:
  • Best Val Accuracy: {df['val_acc'].max()*100:.2f}% (epoch {df['val_acc'].idxmax() + 1})
  • Final Val Accuracy: {df['val_acc'].iloc[-1]*100:.2f}%
  • Final Train Accuracy: {df['train_acc'].iloc[-1]*100:.2f}%

  • Best Val Loss: {df['val_loss'].min():.4f} (epoch {df['val_loss'].idxmin() + 1})
  • Final Val Loss: {df['val_loss'].iloc[-1]:.4f}
  • Final Train Loss: {df['train_loss'].iloc[-1]:.4f}

Training Dynamics:
  • Accuracy Improvement: {(df['val_acc'].iloc[-1] - df['val_acc'].iloc[0])*100:.2f}%
  • Loss Reduction: {(df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]):.4f}
  • Convergence Status: {'✓ Converged' if df['val_loss'].iloc[-1] < df['val_loss'].iloc[-5:].mean() else '⚠ Still improving'}
"""

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        output_path = self.output_dir / 'metrics_summary.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[OK] Metrics summary saved: {output_path}")
        return output_path

    def generate_all(self) -> list:
        """
        Generate all visualization plots for the run.

        Returns:
            List of generated figure paths
        """
        print(f"\n[VISUALIZE] Generating visualizations for run: {self.run_dir.name}")
        print("=" * 60)

        generated = []

        # Generate each visualization type
        curves = self.plot_learning_curves()
        if curves:
            generated.append(curves)

        overfit = self.plot_loss_comparison()
        if overfit:
            generated.append(overfit)

        summary = self.plot_metrics_summary()
        if summary:
            generated.append(summary)

        print("=" * 60)
        print(f"[OK] Generated {len(generated)} visualizations\n")

        return generated


def visualize_run(run_dir: Path, dpi: int = 100) -> list:
    """
    Convenience function to generate all visualizations for a run.

    Usage:
        from src.tools.visualize_run import visualize_run
        visualize_run(Path("runs/official/my_run"))
    """
    visualizer = RunVisualizer(run_dir, dpi=dpi)
    return visualizer.generate_all()


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        run_path = Path(sys.argv[1])
        visualize_run(run_path)
    else:
        print("Usage: python visualize_run.py <run_directory>")
        print("Example: python visualize_run.py runs/official/my_experiment")
