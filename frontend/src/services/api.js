import axios from 'axios';
import { io } from 'socket.io-client';

const API_URL = 'http://localhost:8000/api';

export const getModels = async () => {
  const response = await axios.get(`${API_URL}/models`);
  return response.data.models;
};

export const getModelStructure = async (modelId) => {
  const response = await axios.get(`${API_URL}/models/${modelId}`);
  return response.data;
};

export const getActivations = async (modelId) => {
  const response = await axios.get(`${API_URL}/models/${modelId}/activations`);
  return response.data;
};

export const getGradients = async (modelId) => {
  try {
    const response = await axios.get(`${API_URL}/models/${modelId}/gradients`);
    return response.data;
  } catch (error) {
    console.error(`Failed to get gradients for model ${modelId}:`, error.message);
    if (error.response) {
      if (error.response.status === 404) {
        // Handle 404 - Data not available yet
        return {};  // Return empty object instead of throwing
      }
      if (error.response.status === 500) {
        // Handle 500 - Server error
        console.warn(`Server error when fetching gradients for ${modelId}: ${error.response.data?.error || 'Unknown server error'}`);
        return {};  // Return empty object instead of crashing
      }
    }
    // For network errors or other issues, return empty but don't crash
    return {};
  }
};

export const getStats = async (modelId) => {
  const response = await axios.get(`${API_URL}/models/${modelId}/stats`);
  return response.data;
};

export const connectToSocket = () => {
  const socket = io('http://localhost:8000');
  socket.on('connect', () => {
    console.log('Connected to WebSocket server');
  });
  socket.on('disconnect', () => {
    console.log('Disconnected from WebSocket server');
  });
  return socket;
};
