import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

def create_radar_plot(dataset_name, df):
    categories = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    N = len(categories)
    data = df[df['Dataset'] == dataset_name]
    values_list = []
    labels = data['Model'].tolist()
    for _, row in data.iterrows():
        values = row[categories].tolist()
        values += values[:1]  # repeat first value to close the circle
        values_list.append(values)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    for values, label in zip(values_list, labels):
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)
    plt.title(f'Performance Radar Plot - {dataset_name}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

# Example DataFrame (replace with your real results)
# Removed the 'SAM' model data
df = pd.DataFrame({
    'Dataset': ['Alzheimer']*2,
    'Model': ['CNN', 'Fusion (CNN+SAM)'],
    'Accuracy': [0.80, 0.87],
    'F1-Score': [0.79, 0.86],
    'Precision': [0.78, 0.88],
    'Recall': [0.77, 0.85]
})


create_radar_plot('Alzheimer', df)
