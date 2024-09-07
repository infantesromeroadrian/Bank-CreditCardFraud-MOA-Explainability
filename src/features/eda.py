import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.decorators import timer_decorator, error_handler, log_decorator

class ExploratoryDataAnalysis:
    def __init__(self, data):
        self.data = data

    @timer_decorator
    @error_handler
    @log_decorator
    def plot_histograms(self, columns):
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5 * len(columns)))
        for i, column in enumerate(columns):
            sns.histplot(self.data[column], kde=True, ax=axes[i] if len(columns) > 1 else axes)
            axes[i].set_title(f'Distribución de {column}') if len(columns) > 1 else axes.set_title(f'Distribución de {column}')
        plt.tight_layout()
        plt.show()

    @timer_decorator
    @error_handler
    @log_decorator
    def plot_correlation_matrix(self):
        corr = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Matriz de Correlación')
        plt.show()

    @timer_decorator
    @error_handler
    @log_decorator
    def plot_boxplots(self, columns):
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5 * len(columns)))
        for i, column in enumerate(columns):
            sns.boxplot(x=self.data[column], ax=axes[i] if len(columns) > 1 else axes)
            axes[i].set_title(f'Boxplot de {column}') if len(columns > 1) else axes.set_title(f'Boxplot de {column}')
        plt.tight_layout()
        plt.show()

    @timer_decorator
    @error_handler
    @log_decorator
    def print_summary_statistics(self):
        print(self.data.describe())

    @timer_decorator
    @error_handler
    @log_decorator
    def plot_target_distribution(self, target_column='Class'):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.data[target_column])
        plt.title(f'Distribución de {target_column}')
        plt.show()
