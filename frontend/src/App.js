import React, { useState, useEffect } from 'react';
import { getModelStructure } from './services/api';
import ModelGraph from './components/ModelGraph';
import TrainingMetrics from './components/TrainingMetrics';
import './App.css';

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [connected, setConnected] = useState(true);
  
  // Fetch models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/models');
        const data = await response.json();
        setModels(data.models);
        if (data.models.length > 0 && !selectedModel) {
          setSelectedModel(data.models[0].id);
        }
        setConnected(true);
      } catch (error) {
        console.error('Error fetching models:', error);
        setConnected(false);
      }
    };
    
    fetchModels();
    const interval = setInterval(fetchModels, 5000);
    return () => clearInterval(interval);
  }, [selectedModel]);
  
  // Fetch model structure when selected model changes
  useEffect(() => {
    if (selectedModel) {
      const fetchModelData = async () => {
        try {
          const data = await getModelStructure(selectedModel);
          setModelData(data);
        } catch (error) {
          console.error('Error fetching model data:', error);
        }
      };
      
      fetchModelData();
    }
  }, [selectedModel]);
  
  // Poll for training metrics
  useEffect(() => {
    if (!selectedModel) return;
    
    const fetchTrainingStats = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/models/${selectedModel}/training_stats`);
        if (response.ok) {
          const data = await response.json();
          setTrainingMetrics(data);
        }
      } catch (error) {
        console.error('Error fetching training stats:', error);
      }
    };
    
    fetchTrainingStats(); // Fetch immediately
    
    // Then poll every 500ms
    const interval = setInterval(fetchTrainingStats, 500);
    return () => clearInterval(interval);
  }, [selectedModel]);
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>TensorVision</h1>
        <div className="connection-status">
          {connected ? 
            <span className="status-connected">Connected</span> : 
            <span className="status-disconnected">Disconnected</span>
          }
        </div>
      </header>
      
      <main className="App-main">
        <div className="sidebar">
          <h2>Models</h2>
          {models.length > 0 ? (
            <ul>
              {models.map(model => (
                <li 
                  key={model.id}
                  className={selectedModel === model.id ? 'selected' : ''}
                  onClick={() => setSelectedModel(model.id)}
                >
                  {model.type} ({model.id.substring(0, 8)})
                </li>
              ))}
            </ul>
          ) : (
            <p>No models available. Make sure the backend is running.</p>
          )}
        </div>
        
        <div className="content">
          {!connected && (
            <div className="error-message">
              Cannot connect to backend server. Is it running on port 8000?
            </div>
          )}
          
          {trainingMetrics && (
            <TrainingMetrics metrics={trainingMetrics} />
          )}
          
          {modelData ? (
            <>
              <ModelGraph structure={modelData.structure} />
              <div className="model-info">
                <h3>Model Info</h3>
                <p>Input shape: {JSON.stringify(modelData.input_shape)}</p>
                <p>Output shape: {JSON.stringify(modelData.output_shape)}</p>
              </div>
            </>
          ) : (
            <p>Select a model to visualize</p>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
