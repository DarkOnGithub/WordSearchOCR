import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingDataVisualizer:
    def __init__(self, training_data_path):
        self.training_data_path = Path(training_data_path)
        self.batch_data_path = self.training_data_path / "batch_data"
        self.metrics_path = self.training_data_path / "metrics"
        self.logs_path = self.training_data_path / "logs"

        self.batch_data = pd.DataFrame()
        self.epoch_data = None
        self.additional_metrics = None

    def load_batch_data(self):

        batch_files = sorted(self.batch_data_path.glob("batch_log_epoch_*.txt"))
        all_batches = []

        for batch_file in batch_files:
            epoch_num = int(batch_file.stem.split('_')[-1])
            try:
                df = pd.read_csv(batch_file, sep=',', header=0,
                               names=['batch_idx', 'loss', 'accuracy', 'time_ms'])
                df['epoch'] = epoch_num
                df['global_batch_idx'] = (epoch_num - 1) * len(df) + df['batch_idx']
                all_batches.append(df)
            except Exception as e:
                print(f"Warning: Could not load {batch_file}: {e}")

        if all_batches:
            self.batch_data = pd.concat(all_batches, ignore_index=True)
        else:
            print("No batch data found")

    def load_epoch_data(self):

        epoch_file = self.logs_path / "training_log.txt"
        if epoch_file.exists():
            try:
                self.epoch_data = pd.read_csv(epoch_file, sep=' ', header=None,
                                            names=['epoch', 'train_loss', 'train_accuracy', 'test_accuracy'])
            except Exception as e:
                print(f"Warning: Could not load epoch data: {e}")
        else:
            print("Epoch data file not found")

    def load_additional_metrics(self):

        metrics_file = self.metrics_path / "additional_metrics.txt"
        if metrics_file.exists():
            try:
                self.additional_metrics = pd.read_csv(metrics_file, sep=' ', header=None,
                                                    names=['epoch', 'learning_rate', 'gradient_norm',
                                                           'lr_decay_factor', 'patience_counter'])
            except Exception as e:
                print(f"Warning: Could not load additional metrics: {e}")
        else:
            print("Additional metrics file not found")

    def plot_batch_loss(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.batch_data['global_batch_idx'], self.batch_data['loss'],
                alpha=0.7, linewidth=1, color='blue')
        ax.set_xlabel('Global Batch Index')
        ax.set_ylabel('Loss')
        ax.set_title('Batch-Level Loss')
        ax.grid(True, alpha=0.3)

        max_epoch = self.batch_data['epoch'].max()
        for epoch in range(1, max_epoch + 1):
            epoch_batches = self.batch_data[self.batch_data['epoch'] == epoch]
            if not epoch_batches.empty:
                ax.axvline(x=epoch_batches['global_batch_idx'].max(),
                          linestyle='--', alpha=0.5, color='red')

        plt.tight_layout()
        return fig

    def plot_epoch_loss(self):
        if self.epoch_data is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.epoch_data['epoch'], self.epoch_data['train_loss'],
                marker='o', linewidth=2, color='red', label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Epoch-Level Training Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_loss_distribution(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        epoch_losses = self.batch_data.groupby('epoch')['loss'].agg(['mean', 'std']).reset_index()
        ax.bar(epoch_losses['epoch'], epoch_losses['mean'], yerr=epoch_losses['std'],
               capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.set_title('Average Loss per Epoch with Std Dev')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_loss_vs_learning_rate(self):
        if self.additional_metrics is None or self.epoch_data is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        merged_data = pd.merge(self.epoch_data, self.additional_metrics, on='epoch', how='inner')
        scatter = ax.scatter(merged_data['learning_rate'], merged_data['train_loss'],
                            c=merged_data['epoch'], cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Training Loss')
        ax.set_title('Loss vs Learning Rate (color = epoch)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Epoch')

        plt.tight_layout()
        return fig

    def plot_batch_accuracy(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.batch_data['global_batch_idx'], self.batch_data['accuracy'],
                alpha=0.7, linewidth=1, color='green')
        ax.set_xlabel('Global Batch Index')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Batch-Level Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        return fig

    def plot_epoch_accuracy(self):
        if self.epoch_data is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.epoch_data['epoch'], self.epoch_data['train_accuracy'],
                marker='o', linewidth=2, color='blue', label='Train Accuracy')
        ax.plot(self.epoch_data['epoch'], self.epoch_data['test_accuracy'],
                marker='s', linewidth=2, color='red', label='Test Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Epoch-Level Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)

        plt.tight_layout()
        return fig

    def plot_accuracy_distribution(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        epoch_accuracy = self.batch_data.groupby('epoch')['accuracy'].agg(['mean', 'std']).reset_index()
        ax.bar(epoch_accuracy['epoch'], epoch_accuracy['mean'], yerr=epoch_accuracy['std'],
               capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title('Average Accuracy per Epoch with Std Dev')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        return fig

    def plot_accuracy_gap(self):
        if self.epoch_data is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        accuracy_gap = self.epoch_data['train_accuracy'] - self.epoch_data['test_accuracy']
        colors = ['red' if gap > 5 else 'orange' if gap > 2 else 'green' for gap in accuracy_gap]
        ax.bar(self.epoch_data['epoch'], accuracy_gap, color=colors, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train-Test Accuracy Gap (%)')
        ax.set_title('Overfitting Indicator (Train - Test Gap)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_learning_rate(self):
        if self.additional_metrics is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.additional_metrics['epoch'], self.additional_metrics['learning_rate'],
                marker='o', linewidth=2, color='purple', markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_gradient_norms(self):
        if self.additional_metrics is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.additional_metrics['epoch'], self.additional_metrics['gradient_norm'],
                marker='s', linewidth=2, color='orange', markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_batch_times(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        epoch_times = self.batch_data.groupby('epoch')['time_ms'].agg(['mean', 'std']).reset_index()
        ax.bar(epoch_times['epoch'], epoch_times['mean'], yerr=epoch_times['std'],
               capsize=5, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Batch Time (ms)')
        ax.set_title('Batch Processing Time per Epoch')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_patience_counter(self):
        if self.additional_metrics is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.additional_metrics['epoch'], self.additional_metrics['patience_counter'],
                marker='^', linewidth=2, color='brown', markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Patience Counter')
        ax.set_title('Early Stopping Patience Counter')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_loss_accuracy_trends(self):
        if self.epoch_data is None:
            return

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.plot(self.epoch_data['epoch'], self.epoch_data['train_loss'],
                marker='o', linewidth=2, color='red', label='Train Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.epoch_data['epoch'], self.epoch_data['train_accuracy'],
                     marker='s', linewidth=2, color='blue', label='Train Accuracy', alpha=0.8)
        ax1_twin.plot(self.epoch_data['epoch'], self.epoch_data['test_accuracy'],
                     marker='^', linewidth=2, color='green', label='Test Accuracy', alpha=0.8)
        ax1_twin.set_ylabel('Accuracy (%)', color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        ax1_twin.set_ylim(0, 100)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        ax1.set_title('Loss and Accuracy Trends')

        plt.tight_layout()
        return fig

    def plot_learning_dynamics_combined(self):
        if self.additional_metrics is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.additional_metrics['epoch'], self.additional_metrics['learning_rate'],
                marker='o', linewidth=2, color='purple', label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate', color='purple')
        ax.set_yscale('log')
        ax.tick_params(axis='y', labelcolor='purple')
        ax.grid(True, alpha=0.3)

        ax_twin = ax.twinx()
        ax_twin.plot(self.additional_metrics['epoch'], self.additional_metrics['gradient_norm'],
                     marker='s', linewidth=2, color='orange', label='Gradient Norm')
        ax_twin.set_ylabel('Gradient Norm', color='orange')
        ax_twin.tick_params(axis='y', labelcolor='orange')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.set_title('Learning Dynamics')

        plt.tight_layout()
        return fig

    def plot_recent_loss_distribution(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        recent_epochs = sorted(self.batch_data['epoch'].unique())[-3:]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for i, epoch in enumerate(recent_epochs):
            epoch_data = self.batch_data[self.batch_data['epoch'] == epoch]
            ax.hist(epoch_data['loss'], bins=30, alpha=0.7, color=colors[i],
                    label=f'Epoch {epoch}', density=True)
        ax.set_xlabel('Loss')
        ax.set_ylabel('Density')
        ax.set_title('Loss Distribution (Recent Epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_recent_accuracy_distribution(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        recent_epochs = sorted(self.batch_data['epoch'].unique())[-3:]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for i, epoch in enumerate(recent_epochs):
            epoch_data = self.batch_data[self.batch_data['epoch'] == epoch]
            ax.hist(epoch_data['accuracy'], bins=20, alpha=0.7, color=colors[i],
                    label=f'Epoch {epoch}', density=True)
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Density')
        ax.set_title('Accuracy Distribution (Recent Epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_training_time_analysis(self):
        if self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        epoch_times = self.batch_data.groupby('epoch')['time_ms'].mean()
        ax.bar(epoch_times.index, epoch_times.values, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Batch Time (ms)')
        ax.set_title('Training Time per Epoch')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_early_stopping_monitor(self):
        if self.additional_metrics is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.additional_metrics['epoch'], self.additional_metrics['patience_counter'],
                marker='^', linewidth=2, color='brown', markersize=8)
        ax.fill_between(self.additional_metrics['epoch'], 0, self.additional_metrics['patience_counter'],
                        alpha=0.3, color='brown')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Patience Counter')
        ax.set_title('Early Stopping Monitor')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return fig

    def plot_performance_summary(self):
        if self.epoch_data is None or self.batch_data.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        final_train_acc = self.epoch_data['train_accuracy'].iloc[-1]
        final_test_acc = self.epoch_data['test_accuracy'].iloc[-1]
        final_loss = self.epoch_data['train_loss'].iloc[-1]

        summary_text = f"""
        Training Summary - EMNIST Lowercase Letters OCR

        Final Training Accuracy: {final_train_acc:.2f}%
        Final Test Accuracy: {final_test_acc:.2f}%
        Final Training Loss: {final_loss:.4f}

        Total Epochs: {len(self.epoch_data)}
        Best Test Accuracy: {self.epoch_data['test_accuracy'].max():.2f}% (Epoch {self.epoch_data['test_accuracy'].idxmax() + 1})
        Training Time: ~{(self.batch_data['time_ms'].sum() / 1000 / 60):.1f} minutes

        Total Batches Processed: {len(self.batch_data)}
        Average Batch Loss: {self.batch_data['loss'].mean():.4f}
        Average Batch Accuracy: {self.batch_data['accuracy'].mean():.2f}%
        """

        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Training Summary', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def save_plots(self, output_dir="plots"):
        output_path = Path(self.training_data_path) / output_dir
        output_path.mkdir(exist_ok=True)

        plots = [
            ("batch_loss", self.plot_batch_loss),
            ("epoch_loss", self.plot_epoch_loss),
            ("loss_distribution", self.plot_loss_distribution),
            ("loss_vs_learning_rate", self.plot_loss_vs_learning_rate),

            ("batch_accuracy", self.plot_batch_accuracy),
            ("epoch_accuracy", self.plot_epoch_accuracy),
            ("accuracy_distribution", self.plot_accuracy_distribution),
            ("accuracy_gap", self.plot_accuracy_gap),

            ("learning_rate", self.plot_learning_rate),
            ("gradient_norms", self.plot_gradient_norms),
            ("batch_times", self.plot_batch_times),
            ("patience_counter", self.plot_patience_counter),

            ("loss_accuracy_trends", self.plot_loss_accuracy_trends),
            ("learning_dynamics_combined", self.plot_learning_dynamics_combined),
            ("recent_loss_distribution", self.plot_recent_loss_distribution),
            ("recent_accuracy_distribution", self.plot_recent_accuracy_distribution),
            ("training_time_analysis", self.plot_training_time_analysis),
            ("early_stopping_monitor", self.plot_early_stopping_monitor),
            ("performance_summary", self.plot_performance_summary)
        ]

        saved_files = []
        for plot_name, plot_func in plots:
            try:
                fig = plot_func()
                if fig is not None:
                    filepath = output_path / f"{plot_name}.png"
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_files.append(filepath)
            except Exception as e:
                print(f"Error creating {plot_name} plot: {e}")

        return saved_files

    def print_training_summary(self):
        print("\n" + "="*60)
        print("WORDSEARCHOCR TRAINING SUMMARY")
        print("="*60)

        if self.epoch_data is not None:
            final_epoch = self.epoch_data.iloc[-1]
            best_epoch = self.epoch_data.loc[self.epoch_data['test_accuracy'].idxmax()]

            print("\nFinal Results:")
            print(f"Final Training Accuracy: {final_epoch['train_accuracy']:.2f}%")
            print(f"Final Test Accuracy: {final_epoch['test_accuracy']:.2f}%")
            print(f"Final Training Loss: {final_epoch['train_loss']:.4f}")

            print("\nBest Test Performance:")
            print(f"Best Test Accuracy: {best_epoch['test_accuracy']:.2f}% (Epoch {best_epoch.name + 1})")
            print(f"Training Loss at Best: {best_epoch['train_loss']:.4f}")

        if self.additional_metrics is not None:
            print("\nLearning Dynamics:")
            print(f"Final Learning Rate: {self.additional_metrics['learning_rate'].iloc[-1]:.6f}")
            print(f"Final Gradient Norm: {self.additional_metrics['gradient_norm'].iloc[-1]:.2f}")

        if not self.batch_data.empty:
            print("\nTraining Statistics:")
            print(f"Total batches processed: {len(self.batch_data)}")
            print(f"Total training time: {self.batch_data['time_ms'].sum() / 1000:.1f} seconds")
            print(f"Average batch loss: {self.batch_data['loss'].mean():.4f}")
            print(f"Average batch accuracy: {self.batch_data['accuracy'].mean():.2f}%")

        print("="*60)


def main():
    visualizer = TrainingDataVisualizer("c:/Users/darke/Documents/CLang/WordSearchOCR/training_data")

    visualizer.load_batch_data()
    visualizer.load_epoch_data()
    visualizer.load_additional_metrics()

    visualizer.print_training_summary()

    saved_plots = visualizer.save_plots()
    print(f"\nSuccessfully generated {len(saved_plots)} plots in the training_data/plots/ directory")

    if saved_plots:
        print("\nGenerated plots:")
        for plot in saved_plots:
            print(f"  - {plot.name}")


if __name__ == "__main__":
    main()
