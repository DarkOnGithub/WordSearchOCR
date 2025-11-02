"""
PyTorch implementation of the CNN for letter recognition
Trains on the font-based letter dataset using GPU acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

class LetterCNN(nn.Module):
    """Improved CNN architecture with bottleneck blocks"""

    def __init__(self):
        super(LetterCNN, self).__init__()

        # Block 1: 28x28 -> 14x14 -> 14x14 -> 14x14
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)    # 1->32
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)   # 32->32
        self.conv1c = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)   # 32->64 (1x1)

        # Block 2: 14x14 -> 7x7 -> 7x7 -> 7x7
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)   # 64->64
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)   # 64->64
        self.conv2c = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)  # 64->128 (1x1)

        # Block 3: 7x7 -> 3x3 -> 3x3 -> 3x3
        self.conv3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 128->128
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 128->128
        self.conv3c = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0) # 128->256 (1x1)

        # Max pooling layers (applied after each block)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layers (reduced rates for deeper network)
        self.dropout_conv = nn.Dropout2d(0.15)  # Reduced from 0.25
        self.dropout_fc = nn.Dropout(0.4)       # Reduced from 0.5

        # Fully connected layers (improved capacity distribution)
        # After 3 conv+pool+blocks: 28->14->7->3, so 256*3*3 = 2304
        self.fc1 = nn.Linear(256 * 3 * 3, 128)  # Reduced from 256 to 128
        self.fc2 = nn.Linear(128, 26)           # 26 classes (A-Z)

    def forward(self, x):
        # Block 1: Expansion -> Processing -> Compression
        x = torch.relu(self.conv1a(x))      # 28x28, 1->32
        x = torch.relu(self.conv1b(x))      # 28x28, 32->32
        x = torch.relu(self.conv1c(x))      # 28x28, 32->64
        x = self.pool(x)                    # 28x28 -> 14x14
        x = self.dropout_conv(x)

        # Block 2: Expansion -> Processing -> Compression
        x = torch.relu(self.conv2a(x))      # 14x14, 64->64
        x = torch.relu(self.conv2b(x))      # 14x14, 64->64
        x = torch.relu(self.conv2c(x))      # 14x14, 64->128
        x = self.pool(x)                    # 14x14 -> 7x7
        x = self.dropout_conv(x)

        # Block 3: Expansion -> Processing -> Compression
        x = torch.relu(self.conv3a(x))      # 7x7, 128->128
        x = torch.relu(self.conv3b(x))      # 7x7, 128->128
        x = torch.relu(self.conv3c(x))      # 7x7, 128->256
        x = self.pool(x)                    # 7x7 -> 3x3
        x = self.dropout_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers with improved capacity
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

class LetterDataset(Dataset):
    """PyTorch dataset for letter recognition"""

    def __init__(self, images_path, labels_path):
        # Load numpy arrays
        self.images = np.load(images_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)

        # Labels are already 0-25 (PyTorch convention)
        # No conversion needed

        print(f"Loaded {len(self.images)} samples")
        print(f"Image shape: {self.images.shape}")
        print(f"Label range: {self.labels.min()} to {self.labels.max()}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Add channel dimension and normalize
        image = self.images[idx]
        image = np.expand_dims(image, axis=0)  # Add channel dimension

        # Normalize to [-1, 1] as in C code
        image = (image - 0.5) / 0.5

        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)

def train_model():
    """Train the CNN model"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create model
    model = LetterCNN().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Architecture: 3 bottleneck blocks (3 convs each) -> 2 FC layers")

    # Loss and optimizer (matching C implementation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Default Adam settingsc

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = LetterDataset(
        'data/font_letter_dataset_enhanced/font_letters_train_images.npy',
        'data/font_letter_dataset_enhanced/font_letters_train_labels.npy'
    )
    test_dataset = LetterDataset(
        'data/font_letter_dataset_enhanced/font_letters_test_images.npy',
        'data/font_letter_dataset_enhanced/font_letters_test_labels.npy'
    )

    # Create data loaders
    batch_size = 64  # Matching C implementation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Training parameters
    num_epochs = 10
    best_accuracy = 0.0

    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("\n" + "="*50)
    print("Starting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        epoch_start = time.time()

        # Use tqdm for training progress
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar
            current_loss = running_loss / (train_pbar.n + 1)
            current_acc = 100 * correct_train / total_train
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        train_pbar.close()

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)

        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0

        test_start = time.time()

        # Use tqdm for testing progress
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]", unit="batch")
        with torch.no_grad():
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                # Update test progress
                current_test_acc = 100 * correct_test / total_test
                test_pbar.set_postfix({
                    'acc': f'{current_test_acc:.2f}%'
                })

        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)

        test_time = time.time() - test_start
        epoch_time = time.time() - epoch_start

        test_pbar.close()

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model_pytorch.pth')
            print(f"New best accuracy: {test_accuracy:.2f}%")

        # Early stopping (similar to C implementation)
        patience = 10
        if len(test_accuracies) > patience:
            recent_best = max(test_accuracies[-patience:])
            if test_accuracy < recent_best:
                print(f"No improvement in last {patience} epochs. Early stopping.")
                break

    print("\n" + "="*50)
    print("Training completed!")
    print(".2f")

    # Save final model
    torch.save(model.state_dict(), 'final_model_pytorch.pth')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Progression')
    plt.legend()

    plt.tight_layout()
    plt.savefig('pytorch_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    return model, train_losses, train_accuracies, test_accuracies

def run_inference_on_cells(model_path='best_model_pytorch.pth', cells_dir=r"cells", output_dir='cells_predicted'):
    """Run inference on all cell images and save them with predicted letters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load model
    model = LetterCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Class labels (0-25 -> A-Z)
    class_labels = [chr(ord('A') + i) for i in range(26)]

    # Find all cell images (not the ones with predictions already)
    cell_files = []
    for file in os.listdir(cells_dir):
        if file.startswith('cell_') and file.endswith('.png') and '_pred_' not in file:
            cell_files.append(file)

    print(f"Found {len(cell_files)} cell images to process")

    # Process each image
    processed_count = 0

    for cell_file in tqdm(cell_files, desc="Processing cells"):
        cell_path = os.path.join(cells_dir, cell_file)

        try:
            # Load and preprocess image
            image = Image.open(cell_path).convert('L')  # Convert to grayscale
            image = image.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28

            # Convert to tensor and normalize
            image_array = np.array(image).astype(np.float32)
            image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            image_tensor = (image_tensor / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]  # Get probabilities
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[predicted_class].item() * 100

            predicted_letter = class_labels[predicted_class]

            # Load original image
            original_image = Image.open(cell_path)

            # Save with prediction in filename only (no overlay)
            base_name = os.path.splitext(cell_file)[0]
            output_filename = f"{base_name}_pred_{predicted_letter}_{confidence:.0f}%.png"
            output_path = os.path.join(output_dir, output_filename)

            original_image.save(output_path)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {cell_file}: {e}")
            continue

    print(f"Successfully processed {processed_count} cell images")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if we have a trained model, if not train one
    if os.path.exists('best_model_pytorch.pth'):
        print("Found existing trained model, running inference on cell images...")
        run_inference_on_cells()
    else:
        print("No trained model found, training new model...")

        # Train the model
        model, train_losses, train_accuracies, test_accuracies = train_model()

        # Run inference on cell images
        print("\n" + "="*50)
        print("Running inference on cell images...")
        run_inference_on_cells()
