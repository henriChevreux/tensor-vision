import torch
import torch.nn as nn
import time
import uuid

from .monitor import TensorMonitor

class VisualizableModel(nn.Module):
    """Wrapper for PyTorch models that adds visualization capabilities"""
    
    def __init__(self, model, track_tensors=True, track_gradients=True):
        super().__init__()
        self.model = model
        self.monitor = TensorMonitor()
        self.hooks = {}
        self.track_tensors = track_tensors
        self.track_gradients = track_gradients
        self.model_id = str(uuid.uuid4())
        
        # Register hooks on all modules
        self._register_hooks()
        
        # For handling input/output tensors
        self.latest_input = None
        self.latest_output = None
        
    def _register_hooks(self):
        """Register forward and backward hooks on all modules"""
        for name, module in self.model.named_modules():
            if name == '':  # Skip the root module
                continue
                
            self.hooks[name] = {}
            
            if self.track_tensors:
                self.hooks[name]['forward'] = module.register_forward_hook(
                    self.monitor._forward_hook(name))
                    
            if self.track_gradients:
                self.hooks[name]['backward'] = module.register_backward_hook(
                    self.monitor._backward_hook(name))
    
    def forward(self, *args, **kwargs):
        """Forward pass with timing and input/output tracking"""
        self.latest_input = args[0].detach().cpu() if args and isinstance(args[0], torch.Tensor) else None
        
        start_time = time.time()
        output = self.model(*args, **kwargs)
        end_time = time.time()
        
        self.latest_output = output.detach().cpu() if isinstance(output, torch.Tensor) else None
        self.monitor.module_timing['total'] = end_time - start_time
        
        return output
    
    def get_visualization_data(self):
        """Get all collected data for visualization"""
        return {
            'model_structure': self._get_model_structure(),
            'monitor_data': self.monitor.get_data(),
            'input_shape': tuple(self.latest_input.shape) if self.latest_input is not None else None,
            'output_shape': tuple(self.latest_output.shape) if self.latest_output is not None else None
        }
        
    def _get_model_structure(self):
        """Extract model structure for visualization"""
        structure = []
        
        def add_module(module, name=''):
            # Get basic info about this module
            info = {
                'name': name,
                'type': module.__class__.__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'children': []
            }
            
            # Recursively add children
            for child_name, child_module in module.named_children():
                full_child_name = f"{name}.{child_name}" if name else child_name
                info['children'].append(add_module(child_module, full_child_name))
                
            return info
            
        structure.append(add_module(self.model))
        return structure