from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import threading

class VisualizationServer:
    def __init__(self, port=8000):
        self.port = port
        self.app = Flask(__name__, static_folder='../../frontend/build')
        CORS(self.app)
        self.visual_models = {}
        self.setup_routes()
        
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
        
    def start(self, debug=False, use_thread=True):
        """Start the Flask server"""
        if use_thread:
            threading.Thread(target=self.app.run, 
                            kwargs={'host': '0.0.0.0', 
                                   'port': self.port, 
                                   'debug': debug},
                            daemon=True).start()
        else:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug)