import os
import pandas as pd
import matplotlib.pyplot as plt
TRAINING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
LOGS_PATH = os.path.join(TRAINING_PATH, "logs")

evaluation_logs = pd.read_csv(os.path.join(LOGS_PATH, "evaluation_metrics.csv"))
print(f"Number of Columns: {len(evaluation_logs.columns)}")
print(f"Columns: {evaluation_logs.columns}")

# for column in evaluation_logs.columns:
#     print(f"Column: {column}")
#     print(f"Unique Values: {evaluation_logs[column].unique()}")
#     print(f"Number of Unique Values: {len(evaluation_logs[column].unique())}\n")

episode_losses = evaluation_logs["episode_losses"]
print(f"Type of episode_losses: {type(episode_losses)}") # Series
episode_losses = episode_losses.tolist()
#print(f"Episode Losses: {episode_losses}")
print(f"Number of Values in the Losses array: {len(episode_losses)}") # 50 = number of episodes in a trial

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.title("Episodic Loss Curve")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(range(1, len(episode_losses) + 1), episode_losses, marker='o', linestyle='-', color='crimson')
plt.grid(True)
plt.xticks(ticks=range(0, 51, 5))
plt.tight_layout()
plt.show()