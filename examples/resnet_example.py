import torch
import torchvision.models as models
from tensor_vision import VisualizableModel, VisualizationServer

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Create a visualizable wrapper
visual_model = VisualizableModel(model)

# Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Process the input
with torch.no_grad():
    output = visual_model(dummy_input)

# Create and start the visualization server
server = VisualizationServer(port=8000)
server.register_model(visual_model, "resnet18")
print(f"Starting server at http://localhost:8000")
server.start(use_thread=False)  # Blocking call