import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (adjust if your classes are different)
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Image transform (adjust img_size if needed)
img_size = 1024
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load SAM backbone
try:
    from segment_anything import sam_model_registry
except ImportError:
    print("segment_anything library not found. Please install it: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

sam_checkpoint = 'sam_vit_h_4b8939.pth'  # Update path if needed
if not os.path.exists(sam_checkpoint):
    print(f"SAM checkpoint '{sam_checkpoint}' not found. Please download it and place it in this directory.")
    sys.exit(1)

sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
sam.eval()
sam.to(device)
for param in sam.parameters():
    param.requires_grad = False

def extract_features(images):
    with torch.no_grad():
        feats = sam.image_encoder(images)
        pooled = feats.mean(dim=[2, 3])
    return pooled

# Classifier definition (must match your training code)
class SAMClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

feat_dim = 1024  # For ViT-H
num_classes = 4
model = SAMClassifier(feat_dim, num_classes).to(device)

# Load your trained classifier weights (update the path as needed)
classifier_weights = 'sam_classifier.pth'  # You must provide this file
if not os.path.exists(classifier_weights):
    print(f"Classifier weights '{classifier_weights}' not found. Please provide the trained weights file.")
    sys.exit(1)
model.load_state_dict(torch.load(classifier_weights, map_location=device))
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = extract_features(image)
        logits = model(feats)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    return class_names[pred], confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_path = input("Enter the path to the image: ")
    else:
        image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        sys.exit(1)
    pred_class, conf = predict_image(image_path)
    print(f"Predicted: {pred_class} (Confidence: {conf:.2f})")
