import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=1):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.videos = []
        self.labels = []

        # Load videos from 'real' and 'fake' subfolders
        for label in ['real', 'fake']:
            label_dir = os.path.join(root_dir, label)
            for video_file in os.listdir(label_dir):
                self.videos.append(os.path.join(label_dir, video_file))
                self.labels.append(0 if label == 'real' else 1)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Failed to read {video_path}")

        # Sample frames (e.g., first frame or evenly spaced)
        frame_idx = min(total_frames - 1, total_frames // 2)  # Middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)  # Fallback image

        # Convert to PIL Image for consistency
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if self.transform:
            frame = self.transform(frame)

        return frame, label

class DeepfakeDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=224,
            margin=20,
            device=self.device,
            keep_all=True
        )

        # Initialize and modify ResNet-50
        self.resnet = resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: Real and Fake
        )
        self.resnet = self.resnet.to(device)

        # Define image transformations with augmentation
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_faces(self, image):
        """Detect faces in the image using MTCNN."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes, _ = self.mtcnn.detect(image)
            return boxes if boxes is not None else []
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return []

    def extract_features(self, image, box):
        """Extract features from detected face using ResNet-50."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            face = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            face = self.transform(face).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.resnet(face)
            return features
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return None

    def process_video(self, video_path, output_path=None, frame_sample_rate=5, threshold=0.5):
        """Process video file for deepfake detection."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error opening video file")

            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    cap.get(cv2.CAP_PROP_FPS),
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )

            results = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_sample_rate != 0:  # Sample every nth frame
                    continue

                boxes = self.detect_faces(frame)
                if not boxes:
                    continue

                for box in boxes:
                    features = self.extract_features(frame, box)
                    if features is None:
                        continue

                    prediction = torch.softmax(features, dim=1)
                    prob_fake = prediction[0][1].item()

                    results.append({
                        'frame_number': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        'probability_fake': prob_fake
                    })

                    if output_path:
                        label = f"Fake: {prob_fake:.2f}"
                        color = (0, 0, 255) if prob_fake > threshold else (0, 255, 0)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                                     (int(box[2]), int(box[3])), color, 2)
                        cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if output_path:
                    out.write(frame)

            cap.release()
            if output_path:
                out.release()

            # Video-level prediction
            if results:
                avg_prob_fake = np.mean([r['probability_fake'] for r in results])
                video_label = "Fake" if avg_prob_fake > threshold else "Real"
                return {
                    'video_path': video_path,
                    'avg_probability_fake': avg_prob_fake,
                    'prediction': video_label,
                    'frame_results': results
                }
            return None

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None

    def train(self, train_loader, criterion, optimizer, num_epochs=10):
        """Train the model on a dataset."""
        self.resnet.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.resnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

    def test(self, test_loader):
        """Evaluate the model on a test dataset."""
        self.resnet.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.resnet(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

    def save_model(self, path):
        """Save the trained model."""
        torch.save(self.resnet.state_dict(), path)

    def load_model(self, path):
        """Load a trained model."""
        self.resnet.load_state_dict(torch.load(path, map_location=self.device))
        self.resnet.eval()

# Example usage
if __name__ == "__main__":
    # Paths to dataset (update these based on your local setup)
    dataset_root = "path/to/1000-videos-split"  # Replace with actual path
    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")
    val_dir = os.path.join(dataset_root, "validation")

    # Initialize detector
    detector = DeepfakeDetector()

    # Create datasets and loaders
    train_dataset = DeepfakeDataset(train_dir, transform=detector.transform)
    test_dataset = DeepfakeDataset(test_dir, transform=detector.transform)
    val_dataset = DeepfakeDataset(val_dir, transform=detector.transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.resnet.parameters(), lr=0.001)
    detector.train(train_loader, criterion, optimizer, num_epochs=10)
    detector.save_model("deepfake_model.pth")

    # Evaluation
    detector.test(test_loader)

    # Inference on a single video
    sample_video = os.path.join(test_dir, "fake", "fake_video1.mp4")  # Replace with actual video name
    result = detector.process_video(sample_video, output_path="output_video.mp4")
    if result:
        print(f"Video Prediction: {result['prediction']}, Avg Fake Prob: {result['avg_probability_fake']:.2f}")
