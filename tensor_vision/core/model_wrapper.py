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
        """Extract model structure for visualization in sequential format"""
        def get_module_info(module, name=''):
            # Extract module hierarchy from name
            name_parts = name.split('.')
            module_path = []
            for i in range(len(name_parts)):
                if i % 2 == 0:  # Take every other part as module name
                    module_path.append(name_parts[i])
            
            return {
                'name': name,
                'type': module.__class__.__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'shape': None,  # Will be populated by monitor data if available
                'module_path': module_path,  # Add module hierarchy information
                'module_name': module_path[0] if module_path else 'main'  # Top-level module name
            }

        # Get all modules in order of execution
        modules = []
        input_added = False
        
        for name, module in self.model.named_modules():
            if name == '':  # Skip the root module
                continue
                
            # Skip container modules like Sequential
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue
                
            # Add input layer if not added
            if not input_added:
                input_info = {
                    'name': 'input',
                    'type': 'Input',
                    'parameters': 0,
                    'trainable': 0,
                    'shape': self.latest_input.shape if self.latest_input is not None else None,
                    'module_path': ['input'],
                    'module_name': 'input'
                }
                modules.append(input_info)
                input_added = True
            
            # Add the current module
            module_info = get_module_info(module, name)
            
            # Add shape information if available from monitor
            if name in self.monitor.activations:
                shape_data = self.monitor.activations[name].get('shape', None)
                if shape_data is not None:
                    module_info['shape'] = shape_data
            
            modules.append(module_info)

        # Create sequential structure
        for i in range(len(modules) - 1):
            modules[i]['next'] = modules[i + 1]['name']
            
        # Add output information to the last layer
        if self.latest_output is not None:
            modules[-1]['shape'] = tuple(self.latest_output.shape)
            modules[-1]['module_name'] = 'output'
            modules[-1]['module_path'] = ['output']

        return modules