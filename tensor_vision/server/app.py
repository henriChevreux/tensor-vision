from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import json
import threading

class VisualizationServer:
    def __init__(self, port=8000):
        self.port = port
        self.app = Flask(__name__, static_folder='../../frontend/build')
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
        self.visual_models = {}
        self.latest_stats = {}  # Store latest stats here
        self.setup_routes()
        
        # Add a stats endpoint
        @self.app.route('/api/models/<model_id>/training_stats', methods=['GET'])
        def get_training_stats(model_id):
            if model_id not in self.latest_stats:
                return jsonify({'error': 'No stats available'}), 404
            return jsonify(self.latest_stats[model_id])
        
    def setup_routes(self):
        app = self.app
        
        @app.route('/api/models', methods=['GET'])
        def get_models():
            return jsonify({
                'models': [{
                    'id': model_id,
                    'type': model.model.__class__.__name__
                } for model_id, model in self.visual_models.items()]
            })
            
        @app.route('/api/models/<model_id>', methods=['GET'])
        def get_model_data(model_id):
            if model_id not in self.visual_models:
                return jsonify({'error': 'Model not found'}), 404
                
            # Get model structure only
            data = self.visual_models[model_id].get_visualization_data()
            return jsonify({
                'structure': data['model_structure'],
                'input_shape': data['input_shape'],
                'output_shape': data['output_shape']
            })
            
        @app.route('/api/models/<model_id>/activations', methods=['GET'])
        def get_activations(model_id):
            if model_id not in self.visual_models:
                return jsonify({'error': 'Model not found'}), 404
                
            data = self.visual_models[model_id].get_visualization_data()
            return jsonify(data['monitor_data']['activations'])
            
        @app.route('/api/models/<model_id>/gradients', methods=['GET'])
        def get_gradients(model_id):
            if model_id not in self.visual_models:
                return jsonify({'error': 'Model not found'}), 404
            
            try:
                data = self.visual_models[model_id].get_visualization_data()
                gradients_data = data['monitor_data']['gradients']
                
                # Process gradients data to make it JSON serializable
                processed_gradients = {}
                for layer_name, gradient_info in gradients_data.items():
                    processed_gradients[layer_name] = {
                        'timestamp': gradient_info.get('timestamp', 0),
                        'input': [],
                        'output': []
                    }
                    
                    # Process input gradients
                    if 'input' in gradient_info:
                        for tensor in gradient_info['input']:
                            if tensor is not None:
                                # Get shape and convert data if possible
                                shape = tensor.shape
                                try:
                                    # Convert to Python list for small tensors
                                    if tensor.numel() < 10000:  # Only for reasonably sized tensors
                                        data = tensor.flatten().tolist()
                                    else:
                                        # For large tensors, just provide statistics
                                        data = {
                                            'min': float(tensor.min()),
                                            'max': float(tensor.max()),
                                            'mean': float(tensor.mean()),
                                            'std': float(tensor.std())
                                        }
                                    
                                    processed_gradients[layer_name]['input'].append({
                                        'shape': list(shape),
                                        'data': data
                                    })
                                except Exception as e:
                                    # If conversion fails, just provide shape
                                    processed_gradients[layer_name]['input'].append({
                                        'shape': list(shape) if shape else None,
                                        'error': str(e)
                                    })
                            else:
                                processed_gradients[layer_name]['input'].append(None)
                    
                    # Process output gradients
                    if 'output' in gradient_info:
                        for tensor in gradient_info['output']:
                            if tensor is not None:
                                # Get shape and convert data if possible
                                shape = tensor.shape
                                try:
                                    # Convert to Python list for small tensors
                                    if tensor.numel() < 10000:  # Only for reasonably sized tensors
                                        data = tensor.flatten().tolist()
                                    else:
                                        # For large tensors, just provide statistics
                                        data = {
                                            'min': float(tensor.min()),
                                            'max': float(tensor.max()),
                                            'mean': float(tensor.mean()),
                                            'std': float(tensor.std())
                                        }
                                    
                                    processed_gradients[layer_name]['output'].append({
                                        'shape': list(shape),
                                        'data': data
                                    })
                                except Exception as e:
                                    # If conversion fails, just provide shape
                                    processed_gradients[layer_name]['output'].append({
                                        'shape': list(shape) if shape else None,
                                        'error': str(e)
                                    })
                            else:
                                processed_gradients[layer_name]['output'].append(None)
                
                return jsonify(processed_gradients)
            except Exception as e:
                return jsonify({'error': f'Error processing gradients: {str(e)}'}), 500
            
        @app.route('/api/models/<model_id>/stats', methods=['GET'])
        def get_stats(model_id):
            if model_id not in self.visual_models:
                return jsonify({'error': 'Model not found'}), 404
                
            data = self.visual_models[model_id].get_visualization_data()
            return jsonify(data['monitor_data']['tensor_stats'])
            
        # Serve frontend
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve(path):
            if path != "" and os.path.exists(self.app.static_folder + '/' + path):
                return send_from_directory(self.app.static_folder, path)
            else:
                return send_from_directory(self.app.static_folder, 'index.html')
    
    def register_model(self, model, name=None):
        """Register a model for visualization"""
        from ..core.model_wrapper import VisualizableModel
        
        if not isinstance(model, VisualizableModel):
            model = VisualizableModel(model)
            
        model_id = name or model.model_id
        self.visual_models[model_id] = model
        return model_id

    def emit_training_stats(self, model_id, stats):
        """Emit training statistics to the frontend"""
        # Create application context if needed
        with self.app.app_context():
            self.socketio.emit('trainingStats', {'model_id': model_id, 'stats': stats})
    
    def update_training_stats(self, model_id, stats):
        """Store training statistics for polling"""
        self.latest_stats[model_id] = stats
        
    def start(self, debug=False, use_thread=True):
        """Start the Flask server"""
        if use_thread:
            thread = threading.Thread(target=lambda: self.socketio.run(
                self.app, 
                host='0.0.0.0', 
                port=self.port,
                debug=debug,
                use_reloader=False  # Important - disable reloader
            ))
            thread.daemon = True
            thread.start()
        else:
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)
