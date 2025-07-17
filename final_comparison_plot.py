import matplotlib.pyplot as plt
import pandas as pd

# Updated DataFrame with your new values
df = pd.DataFrame({
    'Dataset': ['Alzheimer']*2,
    'Model': ['CNN', 'Fusion (CNN+SAM)'],
    'Accuracy': [0.82, 0.93],
    'F1-Score': [0.72, 0.92],
    'Precision': [0.73, 0.93],
    'Recall': [0.71, 0.93]
})

# Extracting data for the bar plot
models = df['Model'].tolist()
accuracies = df['Accuracy'].tolist()

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'salmon'])

plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0.70, 1.0)  # Adjusted ylim to better fit the data

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2%}', va='bottom', ha='center')

plt.show()
