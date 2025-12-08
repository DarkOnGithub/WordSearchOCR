"""
PyTorch implementation of the CNN for letter recognition
Trains on the font-based letter dataset using GPU acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os


class LetterCNN_v3_Leaky(nn.Module):
    def __init__(self):
        super(LetterCNN_v3_Leaky, self).__init__()

        # === CHANGED THIS ===
        # Use SiLU (Swish) activation. It's modern and often performs well.
        # Its C implementation is just x * (1 / (1 + exp(-x)))
        self.activation = nn.SiLU()

        # Block 1: 3x3 -> 1x1 (Input: 1x28x28, Output: 64x14x14)
        self.conv1_3x3 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_3x3 = nn.BatchNorm2d(32)
        self.conv1_1x1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.bn1_1x1 = nn.BatchNorm2d(64)
        # === ADDED THIS ===
        # Shortcut to match channel dimensions (1 -> 64) for the residual connection
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64)
        )

        # Block 2: 3x3 -> 1x1 (Input: 64x14x14, Output: 128x7x7)
        self.conv2_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_3x3 = nn.BatchNorm2d(64)
        self.conv2_1x1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn2_1x1 = nn.BatchNorm2d(128)
        # === ADDED THIS ===
        # Shortcut to match channel dimensions (64 -> 128)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

        # Block 3: 3x3 -> 1x1 (Input: 128x7x7, Output: 256x1x1)
        self.conv3_3x3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_3x3 = nn.BatchNorm2d(128)
        self.conv3_1x1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.bn3_1x1 = nn.BatchNorm2d(256)
        # === ADDED THIS ===
        # Shortcut to match channel dimensions (128 -> 256)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )

        # Pooling layers (unchanged)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        self.gap = nn.AdaptiveAvgPool2d((1, 1))           # 7x7 -> 1x1

        # Dropout (unchanged)
        self.dropout_conv = nn.Dropout2d(0.10)
        self.dropout_fc = nn.Dropout(0.25)

        # Fully connected layers (unchanged)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        # --- Block 1: 28x28 -> 14x14 ---
        identity1 = self.shortcut1(x) # Project shortcut

        # Main path
        x_out = self.activation(self.bn1_3x3(self.conv1_3x3(x)))
        x_out = self.bn1_1x1(self.conv1_1x1(x_out)) # <-- NO ACTIVATION HERE

        # Add & Activate
        x = self.activation(x_out + identity1) # <-- APPLY ACTIVATION AFTER ADD

        x = self.pool1(x)
        x = self.dropout_conv(x)

        # --- Block 2: 14x14 -> 7x7 ---
        identity2 = self.shortcut2(x) # Project shortcut

        # Main path
        x_out = self.activation(self.bn2_3x3(self.conv2_3x3(x)))
        x_out = self.bn2_1x1(self.conv2_1x1(x_out)) # <-- NO ACTIVATION HERE

        # Add & Activate
        x = self.activation(x_out + identity2) # <-- APPLY ACTIVATION AFTER ADD

        x = self.pool2(x)
        x = self.dropout_conv(x)

        # --- Block 3: 7x7 -> 1x1 ---
        identity3 = self.shortcut3(x) # Project shortcut

        # Main path
        x_out = self.activation(self.bn3_3x3(self.conv3_3x3(x)))
        x_out = self.bn3_1x1(self.conv3_1x1(x_out)) # <-- NO ACTIVATION HERE

        # Add & Activate
        x = self.activation(x_out + identity3) # <-- APPLY ACTIVATION AFTER ADD

        x = self.gap(x)

        # --- Flatten & FC ---
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
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
    model = LetterCNN_v3_Leaky().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Architecture: LetterCNN_v3_Leaky with BatchNorm, LeakyReLU, and GAP")

    # Loss and optimizer (matching C implementation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)  # Default Adam settingsc

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = LetterDataset(
        'data/font_letters_train_images.npy',
        'data/font_letters_train_labels.npy'
    )
    test_dataset = LetterDataset(
        'data/font_letters_test_images.npy',
        'data/font_letters_test_labels.npy'
    )

    # Create data loaders
    batch_size = 64  # Matching C implementation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Training parameters
    num_epochs = 5
    best_accuracy = 0.0

    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("\n" + "="*50)
    print("Starting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

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
    print(f"Final accuracy: {best_accuracy:.2f}%")

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
    model = LetterCNN_v3_Leaky()
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
            output_filename = f"{predicted_letter}_{np.random.randint(100000, 999999)}.png"
            output_path = os.path.join(output_dir, output_filename)
            original_image.save(output_path)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {cell_file}: {e}")
            continue

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
