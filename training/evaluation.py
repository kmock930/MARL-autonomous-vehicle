import pandas as pd
import matplotlib.pyplot as plt
import os
TRAINING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
from google.protobuf.json_format import MessageToJson
import datetime
from tensorflow.python.profiler import profiler_v2

# Load the saved logs
logs_df = pd.read_csv(os.path.join(TRAINING_PATH, "logs/evaluation_metrics.csv"))

def plot_metrics(logs_df, metric_name, constant_metric, isSaved=False, isShow=False):
    if (metric_name not in logs_df.columns):
        raise ValueError(f"Specific metric '{metric_name}' not found in logs DataFrame.")
    if (metric_name == constant_metric):
        raise ValueError(f"The '{metric_name}' column cannot be plotted as a metric against itself.")

    plt.plot(logs_df[constant_metric], logs_df[metric_name], label=metric_name)
    plt.title(f"{metric_name} over {constant_metric}")
    plt.xlabel(constant_metric)
    plt.ylabel(metric_name)
    plt.legend()
    
    if isSaved:
        output_dir = os.path.join(TRAINING_PATH, "Plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{metric_name}_over_{constant_metric}_plot.png")
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    if isShow:
        plt.show()
    plt.close()

if __name__ == "__main__":
    constant_metrics = ["episode", "reward"]
    for constant_metric in constant_metrics:
        metrics_to_plot = list(set(logs_df.columns) - {constant_metric})

        for metric in metrics_to_plot:
            plot_metrics(
                logs_df=logs_df, 
                metric_name=metric, 
                constant_metric=constant_metric, 
                isSaved=True,
                isShow=False
            )

    # Load Physical Memory Usage
    searchString = "Current process memory usage:"
    LOGS_PATH = os.path.join(TRAINING_PATH, "logs", "train.log")
    memory_usages = [] # per training process
    with open(LOGS_PATH, 'r') as file:
        lines = file.readlines()
        for line in lines:
            for idx, line_content in enumerate(lines):
                if searchString in line_content:
                    # Parsing the memory usage number
                    line_content = line_content.replace(searchString, "")
                    line_content = line_content.replace("MB", "")
                    line_content = line_content.strip()
                    memory_usage = float(line_content)
                    # Print the memory usage
                    memory_usages.append(memory_usage)
                    
    # Plot Memory Usage
    plt.plot(memory_usages)
    plt.title("CPU Memory Usage over Training Process")
    plt.xlabel("Training Process")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    output_dir = os.path.join(TRAINING_PATH, "Plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Memory_Usage_over_Training_Process.png")
    plt.savefig(output_path)
    print(f"CPU Memory Plot has been saved to {output_path}")
    # Average Memory Usage
    avg_memory_usage = sum(memory_usages) / len(memory_usages)
    print(f"Average CPU Memory Usage: {avg_memory_usage:.2f} MB")
