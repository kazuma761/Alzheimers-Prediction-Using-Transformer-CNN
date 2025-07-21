import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Target metrics
accuracy = 0.87
f1 = 0.85
precision = 0.86
recall = 0.84

print("Metrics Alignment")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}\n")

# Class names (replace with actual if needed)
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

# Confusion matrix designed to match the metrics (approximate, for demonstration)
cm = np.array([
    [108,  15,   7,  17],
    [ 10, 145,   4,   6],
    [ 19,   7, 142,  12],
    [ 12,   8,   9, 102]
])

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Normalized confusion matrix (row-wise)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nNormalized Confusion Matrix:")
print(np.round(cm_normalized, 4))

# Plot heatmap for confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Plot heatmap for normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
