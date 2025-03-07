import axios from 'axios';

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
