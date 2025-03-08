import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './ModelGraph.css';

const ModelGraph = ({ structure }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  
  useEffect(() => {
    if (!structure || !structure.length) return;
    
    // Clear previous content
    const container = d3.select(containerRef.current);
    container.selectAll('*').remove();
    
    // Setup dimensions
    const margin = { top: 40, right: 120, bottom: 40, left: 120 };
    const width = 1000;
    const height = 600;
    
    // Create SVG with zoom support
    const svg = container
      .append('svg')
      .attr('width', '100%')
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 2])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);
    
    // Create main group for zoom transform
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create hierarchical layout
    const root = d3.hierarchy(structure[0]);
    
    const treeLayout = d3.tree()
      .size([height - margin.top - margin.bottom, width - margin.left - margin.right]);
    
    treeLayout(root);
    
    // Helper function to get node color based on layer type
    const getNodeColor = (type) => {
      const colors = {
        'Conv2D': '#4CAF50',
        'Dense': '#2196F3',
        'MaxPooling2D': '#9C27B0',
        'Dropout': '#FF9800',
        'Flatten': '#795548',
        'BatchNormalization': '#607D8B',
        'Input': '#E91E63',
        'Output': '#F44336'
      };
      return colors[type] || '#9E9E9E';
    };
    
    // Create curved links
    const links = g.append('g')
      .attr('class', 'links')
      .selectAll('path')
      .data(root.links())
      .enter()
      .append('path')
      .attr('d', d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x))
      .attr('class', 'link');
    
    // Create node groups
    const nodes = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.y},${d.x})`);
    
    // Add node circles
    nodes.append('circle')
      .attr('r', 8)
      .attr('class', 'node-circle')
      .style('fill', d => getNodeColor(d.data.type));
    
    // Add node labels
    nodes.append('text')
      .attr('dy', '0.31em')
      .attr('x', d => d.children ? -12 : 12)
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .text(d => d.data.name)
      .clone(true)
      .lower()
      .attr('stroke', 'white')
      .attr('stroke-width', 3);
    
    // Add tooltips
    const tooltip = container
      .append('div')
      .attr('class', 'model-graph-tooltip')
      .style('opacity', 0);
    
    nodes
      .on('mouseover', (event, d) => {
        tooltip
          .style('opacity', 1)
          .html(`
            <div class="tooltip-title">${d.data.type}</div>
            <div class="tooltip-content">
              <strong>Name:</strong> ${d.data.name}<br/>
              ${d.data.shape ? `<strong>Shape:</strong> ${JSON.stringify(d.data.shape)}<br/>` : ''}
              ${d.data.params ? `<strong>Parameters:</strong> ${d.data.params.toLocaleString()}<br/>` : ''}
              ${d.data.activation ? `<strong>Activation:</strong> ${d.data.activation}` : ''}
            </div>
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mousemove', (event) => {
        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('opacity', 0);
      });
    
    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - margin.right + 20}, ${margin.top})`);
    
    const legendData = Object.entries({
      'Input': '#E91E63',
      'Conv2D': '#4CAF50',
      'Dense': '#2196F3',
      'MaxPooling2D': '#9C27B0',
      'Dropout': '#FF9800',
      'BatchNormalization': '#607D8B',
      'Output': '#F44336'
    });
    
    const legendItems = legend.selectAll('g')
      .data(legendData)
      .enter()
      .append('g')
      .attr('transform', (d, i) => `translate(0, ${i * 25})`);
    
    legendItems.append('circle')
      .attr('r', 6)
      .style('fill', d => d[1]);
    
    legendItems.append('text')
      .attr('x', 15)
      .attr('y', 4)
      .text(d => d[0]);
    
  }, [structure]);
  
  return (
    <div className="model-graph-container" ref={containerRef}>
      <div className="model-graph-controls">
        <button onClick={() => {
          const svg = d3.select(containerRef.current).select('svg');
          svg.call(d3.zoom().transform, d3.zoomIdentity);
        }}>Reset Zoom</button>
      </div>
    </div>
  );
};

export default ModelGraph;
