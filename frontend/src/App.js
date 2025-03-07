import React, { useState, useEffect } from 'react';
import { getModels, getModelStructure } from './services/api';
import ModelGraph from './components/ModelGraph';
import './App.css';

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelData, setModelData] = useState(null);
  
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels();
        setModels(data);
        if (data.length > 0) {
          setSelectedModel(data[0].id);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };
    
    fetchModels();
    
    // Poll for new models every 5 seconds
    const interval = setInterval(fetchModels, 5000);
    return () => clearInterval(interval);
  }, []);
  
  useEffect(() => {
    if (!selectedModel) return;
    
    const fetchModelData = async () => {
      try {
        const data = await getModelStructure(selectedModel);
        setModelData(data);
      } catch (error) {
        console.error('Error fetching model data:', error);
      }
    };
    
    fetchModelData();
  }, [selectedModel]);
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>TensorVision</h1>
      </header>
      
      <main className="App-main">
        <div className="sidebar">
          <h2>Models</h2>
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
        </div>
        
        <div className="content">
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
