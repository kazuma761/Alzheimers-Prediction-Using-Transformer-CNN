import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Configuration - Update these paths according to your setup
data_dir = r"C:\Users\shrir\Music\New folder\Data"
model_path = r"C:\Users\shrir\Music\New folder\hello_new_cnn_model.h5"

# Known Performance Metrics (from your results)
KNOWN_METRICS = {
    'Accuracy': 0.87,
    'F1-Score': 0.85,
    'Precision': 0.86,
    'Recall': 0.84
}

# Parameters
img_height, img_width = 128, 128
batch_size = 32

print("="*70)
print("OASIS DEMENTIA CLASSIFICATION - CONFUSION MATRIX ANALYSIS")
print("="*70)
print(f"Expected Performance Metrics:")
for metric, value in KNOWN_METRICS.items():
    print(f"  {metric}: {value:.2f}")
print("="*70)

# Load the pre-trained model
model = load_model(model_path)
print("✓ Model loaded successfully!")

# Create validation data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important: Don't shuffle for consistent results
)

print(f"✓ Validation samples: {val_gen.samples}")
print(f"✓ Class indices: {val_gen.class_indices}")

# Get predictions
print("\nGenerating predictions...")
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get true labels
y_true = val_gen.classes

# Get class names
class_names = list(val_gen.class_indices.keys())
print(f"✓ Classes: {class_names}")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate actual metrics
actual_accuracy = accuracy_score(y_true, y_pred)
actual_f1 = f1_score(y_true, y_pred, average='weighted')
actual_precision = precision_score(y_true, y_pred, average='weighted')
actual_recall = recall_score(y_true, y_pred, average='weighted')

print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
print(cm)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# Create a 2x2 subplot layout
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

# Plot 1: Confusion Matrix with counts (top-left)
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax1,
            cbar_kws={'label': 'Number of Samples'})
ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)

# Plot 2: Normalized Confusion Matrix (top-right)
ax2 = fig.add_subplot(gs[0, 1])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names, ax=ax2,
            cbar_kws={'label': 'Percentage'})
ax2.set_title('Normalized Confusion Matrix (Percentages)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)

# Plot 3: Metrics Comparison (bottom spanning both columns)
ax3 = fig.add_subplot(gs[1, :])

# Prepare data for comparison
metrics_comparison = pd.DataFrame({
    'Expected': [KNOWN_METRICS['Accuracy'], KNOWN_METRICS['F1-Score'], 
                KNOWN_METRICS['Precision'], KNOWN_METRICS['Recall']],
    'Actual': [actual_accuracy, actual_f1, actual_precision, actual_recall]
}, index=['Accuracy', 'F1-Score', 'Precision', 'Recall'])

# Create bar plot
x = np.arange(len(metrics_comparison.index))
width = 0.35

bars1 = ax3.bar(x - width/2, metrics_comparison['Expected'], width, 
                label='Expected', color='skyblue', alpha=0.8)
bars2 = ax3.bar(x + width/2, metrics_comparison['Actual'], width,
                label='Actual', color='lightcoral', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

ax3.set_xlabel('Metrics', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Expected vs Actual Performance Metrics', fontsize=16, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics_comparison.index)
ax3.legend()
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confusion_matrix_oasis_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate detailed classification report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df.round(4))

# Calculate and display per-class accuracy
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("="*70)
for i, class_name in enumerate(class_names):
    class_accuracy = cm[i, i] / cm[i, :].sum()
    class_precision = report[class_name]['precision']
    class_recall = report[class_name]['recall']
    class_f1 = report[class_name]['f1-score']
    
    print(f"{class_name}:")
    print(f"  Accuracy:  {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    print(f"  Precision: {class_precision:.4f} ({class_precision*100:.2f}%)")
    print(f"  Recall:    {class_recall:.4f} ({class_recall*100:.2f}%)")
    print(f"  F1-Score:  {class_f1:.4f} ({class_f1*100:.2f}%)")
    print()

# Metrics comparison
print("="*70)
print("EXPECTED vs ACTUAL METRICS COMPARISON")
print("="*70)
print(f"{'Metric':<12} {'Expected':<10} {'Actual':<10} {'Difference':<12}")
print("-" * 50)
print(f"{'Accuracy':<12} {KNOWN_METRICS['Accuracy']:<10.3f} {actual_accuracy:<10.3f} {abs(KNOWN_METRICS['Accuracy'] - actual_accuracy):<12.3f}")
print(f"{'F1-Score':<12} {KNOWN_METRICS['F1-Score']:<10.3f} {actual_f1:<10.3f} {abs(KNOWN_METRICS['F1-Score'] - actual_f1):<12.3f}")
print(f"{'Precision':<12} {KNOWN_METRICS['Precision']:<10.3f} {actual_precision:<10.3f} {abs(KNOWN_METRICS['Precision'] - actual_precision):<12.3f}")
print(f"{'Recall':<12} {KNOWN_METRICS['Recall']:<10.3f} {actual_recall:<10.3f} {abs(KNOWN_METRICS['Recall'] - actual_recall):<12.3f}")

# Additional insights
print("\n" + "="*70)
print("CONFUSION MATRIX INSIGHTS")
print("="*70)

# Find most confused classes
max_confusion = 0
most_confused_pair = None
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > max_confusion:
            max_confusion = cm[i, j]
            most_confused_pair = (class_names[i], class_names[j])

if most_confused_pair:
    print(f"Most confused classes: {most_confused_pair[0]} → {most_confused_pair[1]} ({max_confusion} cases)")

# Best performing class
class_accuracies = [cm[i, i] / cm[i, :].sum() for i in range(len(class_names))]
best_class_idx = np.argmax(class_accuracies)
best_class_accuracy = class_accuracies[best_class_idx]
print(f"Best performing class: {class_names[best_class_idx]} ({best_class_accuracy*100:.2f}% accuracy)")

# Worst performing class
worst_class_idx = np.argmin(class_accuracies)
worst_class_accuracy = class_accuracies[worst_class_idx]
print(f"Worst performing class: {class_names[worst_class_idx]} ({worst_class_accuracy*100:.2f}% accuracy)")

# Class distribution analysis
print(f"\nClass distribution in validation set:")
for i, class_name in enumerate(class_names):
    class_count = cm[i, :].sum()
    class_percentage = (class_count / len(y_true)) * 100
    print(f"  {class_name}: {class_count} samples ({class_percentage:.1f}%)")

print(f"\nTotal validation samples: {len(y_true)}")
print("\n✓ Enhanced confusion matrix visualization saved as 'confusion_matrix_oasis_enhanced.png'")
print("✓ Analysis complete!")