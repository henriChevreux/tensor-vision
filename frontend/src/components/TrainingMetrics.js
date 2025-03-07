import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const TrainingMetrics = ({ metrics }) => {
  const [chartData, setChartData] = useState([]);
  const [historyData, setHistoryData] = useState([]);
  
  // Initialize or update history when metrics change
  useEffect(() => {
    if (!metrics) return;
    
    // Add current metrics to history
    setHistoryData(prevHistory => {
      // Create a copy of the previous history
      const newHistory = [...prevHistory];
      
      // Add new data point
      newHistory.push({
        timestamp: metrics.timestamp ? metrics.timestamp * 1000 : Date.now(),
        loss: metrics.loss,
        accuracy: metrics.accuracy,
        epoch: metrics.epoch,
        batch: metrics.batch
      });
      
      // Limit history size to avoid performance issues
      if (newHistory.length > 100) {
        return newHistory.slice(-100);
      }
      
      return newHistory;
    });
  }, [metrics]);
  
  // Update chart data when history changes
  useEffect(() => {
    // Format history data for the chart
    const formattedData = historyData.map((item, index) => ({
      index,
      loss: item.loss,
      accuracy: item.accuracy / 100, // Scale to 0-1 range for better visualization
      label: `E${item.epoch}B${item.batch}`
    }));
    
    setChartData(formattedData);
  }, [historyData]);
  
  if (!metrics) {
    return <div className="metrics-placeholder">Waiting for training metrics...</div>;
  }
  
  return (
    <div className="training-metrics">
      <h3>Training Metrics</h3>
      
      <div className="current-metrics">
        <div className="metric-card">
          <h4>Loss</h4>
          <div className="metric-value">{metrics.loss?.toFixed(4) || 'N/A'}</div>
        </div>
        <div className="metric-card">
          <h4>Accuracy</h4>
          <div className="metric-value">{metrics.accuracy?.toFixed(2) || 'N/A'}%</div>
        </div>
        <div className="metric-card">
          <h4>Epoch / Batch</h4>
          <div className="metric-value">{metrics.epoch || 'N/A'} / {metrics.batch || 'N/A'}</div>
        </div>
      </div>
      
      <div className="metrics-chart">
        <h4>Training Progress</h4>
        {chartData.length > 1 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis 
                yAxisId="left" 
                orientation="left" 
                domain={[0, dataMax => Math.max(dataMax * 1.2, 0.5)]}
              />
              <YAxis 
                yAxisId="right" 
                orientation="right" 
                domain={[0, 1.1]} 
              />
              <Tooltip 
                formatter={(value, name) => {
                  if (name === 'accuracy') {
                    return [(value * 100).toFixed(2) + '%', 'Accuracy'];
                  }
                  return [value.toFixed(4), name === 'loss' ? 'Loss' : name];
                }}
              />
              <Legend />
              <Line 
                yAxisId="left" 
                type="monotone" 
                dataKey="loss" 
                stroke="#8884d8" 
                name="Loss" 
                dot={false}
                activeDot={{ r: 6 }}
              />
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#82ca9d" 
                name="Accuracy" 
                dot={false}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="chart-placeholder">
            Collecting data for chart visualization...
          </div>
        )}
      </div>
      
      <div className="layer-activations">
        <h4>Recent Activity</h4>
        <div className="activity-log">
          {historyData.slice(-5).reverse().map((item, index) => (
            <div key={index} className="activity-item">
              <span className="timestamp">
                {new Date(item.timestamp).toLocaleTimeString()}
              </span>
              <span className="details">
                Epoch {item.epoch}, Batch {item.batch}: 
                Loss: {item.loss.toFixed(4)}, 
                Accuracy: {item.accuracy.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TrainingMetrics;