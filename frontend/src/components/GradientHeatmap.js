import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GradientHeatmap.css';

const GradientHeatmap = ({ gradients }) => {
  const containerRef = useRef();
  const svgRef = useRef(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [availableLayers, setAvailableLayers] = useState([]);
  const [error, setError] = useState(null);

  // Handle data changes and layer selection
  useEffect(() => {
    // Reset error state on new data
    setError(null);
    
    // Check if gradients data is valid
    if (!gradients || Object.keys(gradients).length === 0) {
      setError("No gradient data available");
      return;
    }
    
    // Extract available layers
    const layers = Object.keys(gradients);
    setAvailableLayers(layers);
    
    // Select first layer by default if none selected
    if (!selectedLayer && layers.length > 0) {
      setSelectedLayer(layers[0]);
    }
  }, [gradients, selectedLayer]);

  // Render visualization
  useEffect(() => {
    // Safely clean up any existing visualization
    const container = d3.select(containerRef.current);
    container.selectAll('svg').remove();
    
    // Reset error state
    setError(null);

    // Check if we have valid selected layer and gradient data
    if (!selectedLayer || !gradients || !gradients[selectedLayer]) {
      setError("No layer selected or gradient data not available");
      return;
    }
    
    const gradientData = gradients[selectedLayer];
    
    // Check if we have output gradient data
    if (!gradientData.output || !gradientData.output.length || 
        !gradientData.output[0]) {
      setError(`No gradient data available for layer: ${selectedLayer}`);
      return;
    }
    
    try {
      // Get first output gradient
      const outputGradient = gradientData.output[0];
      
      // Extract gradient data based on the format
      let gradientValues = [];
      let shapeInfo = null;
      
      // Handle different data formats from backend
      if (outputGradient && outputGradient.data) {
        if (Array.isArray(outputGradient.data)) {
          // Direct array of values
          gradientValues = outputGradient.data;
        } else if (typeof outputGradient.data === 'object' && 
                  'min' in outputGradient.data && 
                  'max' in outputGradient.data) {
          // Statistics format for large tensors
          setError(`Tensor too large to visualize. Statistics: Min=${outputGradient.data.min.toFixed(6)}, Max=${outputGradient.data.max.toFixed(6)}, Mean=${outputGradient.data.mean.toFixed(6)}`);
          return;
        }
        
        // Get shape information if available
        if (outputGradient.shape) {
          shapeInfo = outputGradient.shape;
        }
      } else if (Array.isArray(outputGradient)) {
        // Direct array format
        gradientValues = outputGradient;
      } else {
        setError(`Unsupported gradient data format for layer: ${selectedLayer}`);
        return;
      }
      
      // Ensure we have data to visualize
      if (!gradientValues || gradientValues.length === 0) {
        setError(`No gradient values available for layer: ${selectedLayer}`);
        return;
      }
      
      // Ensure values are numeric
      if (typeof gradientValues[0] !== 'number') {
        setError(`Gradient data is not numeric for layer: ${selectedLayer}`);
        return;
      }
      
      // Get dimensions for visualization
      const width = 600;
      const height = 400;
      const padding = 40;
      
      // Create SVG
      const svg = container
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');
      
      svgRef.current = svg.node();
      
      // Calculate min and max values for color scale
      const minValue = d3.min(gradientValues);
      const maxValue = d3.max(gradientValues);
      
      // Create a symmetric diverging color scale from blue (negative) to white (zero) to red (positive)
      const absMax = Math.max(Math.abs(minValue), Math.abs(maxValue), 0.000001); // Avoid zero max value
      
      const colorScale = d3.scaleSequential()
        .domain([-absMax, absMax])
        .interpolator(d3.interpolateRdBu);
      
      // Get the dimensions of the data
      let numRows, numCols;
      
      if (shapeInfo && shapeInfo.length >= 2) {
        // For 2D or higher tensors
        numRows = shapeInfo[0];
        numCols = shapeInfo[1];
      } else {
        // For 1D tensors, create a square-ish representation
        const total = gradientValues.length;
        numCols = Math.ceil(Math.sqrt(total));
        numRows = Math.ceil(total / numCols);
      }
      
      // Calculate cell size
      const cellWidth = (width - 2 * padding) / numCols;
      const cellHeight = (height - 2 * padding) / numRows;
      
      // Create cells - using a single group for all cells
      const cellGroup = svg.append('g')
        .attr('transform', `translate(${padding}, ${padding})`);
      
      // Add cells as individual rectangles
      gradientValues.forEach((value, i) => {
        const x = (i % numCols) * cellWidth;
        const y = Math.floor(i / numCols) * cellHeight;
        
        const cell = cellGroup.append('rect')
          .attr('x', x)
          .attr('y', y)
          .attr('width', cellWidth - 1)
          .attr('height', cellHeight - 1)
          .attr('fill', isNaN(value) || !isFinite(value) ? '#cccccc' : colorScale(value))
          .attr('stroke', '#e0e0e0')
          .attr('stroke-width', 0.5);
        
        // Add tooltip
        cell.append('title')
          .text(isNaN(value) || !isFinite(value) ? 
                'Invalid value' : 
                `Gradient value: ${Math.abs(value) < 0.001 ? value.toExponential(4) : value.toFixed(6)}`);
      });
      
      // Add color legend
      const legendWidth = width - 2 * padding;
      const legendHeight = 20;
      
      const legendX = padding;
      const legendY = height - padding;
      
      // Create gradient for legend
      const defs = svg.append('defs');
      const linearGradient = defs.append('linearGradient')
        .attr('id', `gradient-legend-${selectedLayer.replace(/[^a-zA-Z0-9]/g, '-')}`)
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '0%');
      
      // Add color stops
      linearGradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', colorScale(-absMax));
        
      linearGradient.append('stop')
        .attr('offset', '50%')
        .attr('stop-color', colorScale(0));
        
      linearGradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', colorScale(absMax));
      
      // Create legend rectangle
      svg.append('rect')
        .attr('x', legendX)
        .attr('y', legendY)
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', `url(#gradient-legend-${selectedLayer.replace(/[^a-zA-Z0-9]/g, '-')})`);
      
      // Add min, zero, and max labels
      svg.append('text')
        .attr('x', legendX)
        .attr('y', legendY + legendHeight + 15)
        .attr('text-anchor', 'start')
        .text(Math.abs(minValue) < 0.001 ? minValue.toExponential(2) : minValue.toFixed(4));
      
      svg.append('text')
        .attr('x', legendX + legendWidth / 2)
        .attr('y', legendY + legendHeight + 15)
        .attr('text-anchor', 'middle')
        .text('0');
      
      svg.append('text')
        .attr('x', legendX + legendWidth)
        .attr('y', legendY + legendHeight + 15)
        .attr('text-anchor', 'end')
        .text(Math.abs(maxValue) < 0.001 ? maxValue.toExponential(2) : maxValue.toFixed(4));
      
      // Add title
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', padding / 2)
        .attr('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text(`Gradient Heatmap for ${selectedLayer}`);
    } catch (e) {
      console.error('Error rendering gradient heatmap:', e);
      setError(`Error rendering heatmap: ${e.message}`);
    }
    
    // Cleanup function
    return () => {
      if (svgRef.current && svgRef.current.parentNode) {
        svgRef.current.parentNode.removeChild(svgRef.current);
      }
    };
  }, [selectedLayer, gradients]);

  const handleLayerChange = (e) => {
    // Safely select a new layer
    setSelectedLayer(e.target.value);
  };

  return (
    <div className="gradient-heatmap-container">
      <div className="layer-selector">
        <label htmlFor="layer-select">Select Layer: </label>
        <select 
          id="layer-select"
          value={selectedLayer || ''}
          onChange={handleLayerChange}
          disabled={availableLayers.length === 0}
        >
          <option value="" disabled>Select a layer</option>
          {availableLayers.map(layer => (
            <option key={layer} value={layer}>{layer}</option>
          ))}
        </select>
      </div>
      <div className="heatmap-visualization" ref={containerRef}>
        {error && (
          <div className="no-data-message">{error}</div>
        )}
      </div>
    </div>
  );
};

export default GradientHeatmap; 
