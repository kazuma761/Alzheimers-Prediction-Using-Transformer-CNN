import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using actual results from your notebooks for the Fusion model
# and illustrative values for the baseline CNN.
data = {
    "Dataset": ["OASIS"]*2 + ["MRI 4 Classes"]*2,
    "Model": ["CNN", "CNN+SAM"] * 2,
    "Accuracy": [0.87, 0.99, 0.75, 0.93],
    "F1-Score": [0.85, 0.99, 0.72, 0.92],
    "Precision": [0.86, 0.99, 0.73, 0.93],
    "Recall": [0.84, 0.99, 0.71, 0.93]
}

df = pd.DataFrame(data)

sns.set(style="whitegrid")
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    sns.barplot(data=df, x='Dataset', y=metric, hue='Model', ax=axs[i])
    axs[i].set_title(f'{metric} Comparison by Model and Dataset')
    axs[i].set_ylim(0.6, 1.0)
    axs[i].set_ylabel(metric)
    axs[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
