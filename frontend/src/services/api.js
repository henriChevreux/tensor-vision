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
