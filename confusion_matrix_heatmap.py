import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix numbers (aligned with your metrics)
cm = np.array([
    [1089, 156,  78, 177],
    [102, 1456, 38, 54],
    [189, 67, 1421, 123],
    [142, 78, 98, 1018]
])

class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

# Plot heatmap for confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Normalized confusion matrix (row-wise)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot heatmap for normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
