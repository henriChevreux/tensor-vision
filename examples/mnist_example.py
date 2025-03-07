import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensor_vision import VisualizableModel, VisualizationServer
import time

# Define a simple CNN for MNIST
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Load MNIST dataset
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    
    return train_loader

def train_model(model, train_loader, server, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device}...")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            
            # Update running loss
            running_loss += loss.item()
            
            # Update training statistics (now using the safer method)
            if i % 10 == 0:  # Update every 10 batches
                stats = {
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'loss': loss.item(),
                    'accuracy': accuracy,
                    'running_loss': running_loss / (i + 1),
                    'timestamp': time.time()
                }
                server.update_training_stats('mnist_cnn', stats)
            
            # Print statistics
            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f} Accuracy: {accuracy:.2f}%')
                running_loss = 0.0
                
    print('Finished Training')
    return model

if __name__ == "__main__":
    # Create model
    model = MnistCNN()
    
    # Wrap with visualization
    visual_model = VisualizableModel(model)
    
    # Create and start visualization server in a background thread
    server = VisualizationServer(port=8000)
    server.register_model(visual_model, "mnist_cnn")
    server.start(use_thread=True, debug=False)
    
    print(f"Visualization server running at http://localhost:8000")
    print("Training model...")
    
    # Load data and train model
    train_loader = load_data()
    train_model(visual_model, train_loader, server, num_epochs=3)
    
    print("Training complete. Keeping server alive for visualization.")
    print("Press Ctrl+C to exit.")
    
    # Keep the script running to allow exploration of the trained model
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")
