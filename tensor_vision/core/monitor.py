import torch
import threading
import time
from collections import defaultdict

class TensorMonitor:
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.module_timing = {}
        self.tensor_shapes = {}
        self.tensor_stats = {}
        self._lock = threading.Lock()
        
    def _forward_hook(self, name):
        def hook(module, input_tensor, output_tensor):
            with self._lock:
                # Store activations
                self.activations[name] = {
                    'data': output_tensor.detach().cpu().clone() if isinstance(output_tensor, torch.Tensor) 
                           else [t.detach().cpu().clone() for t in output_tensor if isinstance(t, torch.Tensor)],
                    'timestamp': time.time()
                }
                
                # Calculate basic statistics if possible
                if isinstance(output_tensor, torch.Tensor):
                    tensor = output_tensor.detach().cpu()
                    self.tensor_stats[name] = {
                        'min': float(tensor.min()),
                        'max': float(tensor.max()),
                        'mean': float(tensor.mean()),
                        'std': float(tensor.std()),
                        'has_nan': bool(torch.isnan(tensor).any()),
                        'has_inf': bool(torch.isinf(tensor).any())
                    }
        return hook
    
    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            with self._lock:
                # Store gradients
                self.gradients[name] = {
                    'input': [x.detach().cpu().clone() if isinstance(x, torch.Tensor) and x is not None else None 
                             for x in grad_input],
                    'output': [x.detach().cpu().clone() if isinstance(x, torch.Tensor) and x is not None else None 
                              for x in grad_output],
                    'timestamp': time.time()
                }
        return hook
    
    def get_data(self):
        """Return collected data for visualization"""
        with self._lock:
            return {
                'activations': {k: {
                    'shape': v['data'].shape if isinstance(v['data'], torch.Tensor) else 
                            [t.shape for t in v['data'] if isinstance(t, torch.Tensor)],
                    'timestamp': v['timestamp']
                } for k, v in self.activations.items()},
                'gradients': self.gradients,
                'tensor_shapes': self.tensor_shapes,
                'tensor_stats': self.tensor_stats,
                'module_timing': self.module_timing
            }