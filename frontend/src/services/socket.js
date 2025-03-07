import { io } from 'socket.io-client';

class SocketService {
  constructor() {
    this.socket = null;
    this.callbacks = {};
  }

  connect(url) {
    console.log(`Connecting to Socket.IO at ${url}`);
    this.socket = io(url);

    this.socket.on('connect', () => {
      console.log('Socket.IO connected');
      if (this.callbacks.onConnect) {
        this.callbacks.onConnect();
      }
    });

    this.socket.on('disconnect', () => {
      console.log('Socket.IO disconnected');
      if (this.callbacks.onDisconnect) {
        this.callbacks.onDisconnect();
      }
    });

    this.socket.on('trainingStats', (data) => {
      console.log('Received training stats:', data);
      if (this.callbacks.trainingStats) {
        this.callbacks.trainingStats(data);
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error);
      if (this.callbacks.onError) {
        this.callbacks.onError(error);
      }
    });
  }

  on(event, callback) {
    this.callbacks[event] = callback;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
    }
  }
}

const socketService = new SocketService();
export default socketService;