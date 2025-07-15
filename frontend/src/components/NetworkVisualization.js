import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './NetworkVisualization.css';

const NetworkVisualization = ({ structure }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const [viewMode, setViewMode] = useState('2d');
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const [dragging, setDragging] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [connectionDensity, setConnectionDensity] = useState('sequential'); // 'sequential', 'layerwise', 'full'
  
  // Function to toggle between 2D and 3D views
  const toggleViewMode = () => {
    setViewMode(prev => prev === '2d' ? '3d' : '2d');
    setRotation({ x: 0, y: 0, z: 0 }); // Reset rotation when toggling
  };

  // Function to cycle through connection densities
  const cycleConnectionDensity = () => {
    setConnectionDensity(prev => {
      switch(prev) {
        case 'sequential': return 'layerwise';
        case 'layerwise': return 'full';
        case 'full': return 'sequential';
        default: return 'sequential';
      }
    });
  };

  // Handle mouse down for dragging
  const handleMouseDown = (event) => {
    if (viewMode === '3d') {
      setDragging(true);
      setLastPos({ x: event.clientX, y: event.clientY });
    }
  };

  // Handle mouse move for dragging
  const handleMouseMove = (event) => {
    if (dragging && viewMode === '3d') {
      // Calculate the delta movement
      const deltaX = event.clientX - lastPos.x;
      const deltaY = event.clientY - lastPos.y;
      
      // Update rotation based on mouse movement
      setRotation(prev => ({
        x: prev.x + deltaY * 0.5, // Vertical movement rotates around X-axis
        y: prev.y + deltaX * 0.5, // Horizontal movement rotates around Y-axis
        z: prev.z
      }));
      
      setLastPos({ x: event.clientX, y: event.clientY });
    }
  };

  // Handle mouse up to stop dragging
  const handleMouseUp = () => {
    setDragging(false);
  };

  // Add event listeners for drag interaction
  useEffect(() => {
    const container = containerRef.current;
    if (container && viewMode === '3d') {
      container.addEventListener('mousedown', handleMouseDown);
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        container.removeEventListener('mousedown', handleMouseDown);
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [viewMode, dragging, lastPos]);

  useEffect(() => {
    if (!structure || !structure.length) {
      console.log("No structure data available");
      return;
    }
    
    console.log("Structure data:", structure);

    // Clear previous visualizations
    const container = d3.select(containerRef.current);
    container.selectAll('svg').remove();
    container.selectAll('.network-tooltip').remove();
    container.selectAll('.view-controls').remove();

    // Setup dimensions
    const width = 800;
    const height = 600;
    const margin = { top: 50, right: 50, bottom: 50, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    svgRef.current = svg.node();

    // Add a fancy background gradient
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'network-bg-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '100%')
      .attr('y2', '100%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#111827')
      .attr('stop-opacity', 1);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#1f2937')
      .attr('stop-opacity', 1);

    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'url(#network-bg-gradient)');

    // Process nodes to set positions in a layout
    const processedNodes = [];
    const links = [];
    
    // Process nodes to set positions based on the structure
    structure.forEach((node, i) => {
      // Extract useful properties from the node
      const name = node.name || node.module_name || `Node ${i}`;
      const type = node.module_name || "Unknown";
      const params = parseInt(node.parameters || 0);
      const next = node.next || null;
      
      processedNodes.push({
        id: i,
        x: 0, // Will set positions later
        y: 0, // Will set positions later
        z: 0, // Z position for 3D
        name: name,
        type: type,
        parameters: params,
        next: next,
        index: i
      });
    });
    
    // Add tooltip container
    const tooltip = container
      .append('div')
      .attr('class', 'network-tooltip')
      .style('opacity', 0);
    
    // Add 3D visualization controls
    const controls = container
      .append('div')
      .attr('class', 'view-controls')
      .style('position', 'absolute')
      .style('top', '10px')
      .style('right', '10px');
    
    controls.append('button')
      .attr('class', 'view-toggle-btn')
      .text(viewMode === '2d' ? 'Switch to 3D' : 'Switch to 2D')
      .on('click', toggleViewMode);
      
    // Add connection density control
    controls.append('button')
      .attr('class', 'connection-toggle-btn')
      .text(`Connections: ${connectionDensity.charAt(0).toUpperCase() + connectionDensity.slice(1)}`)
      .style('margin-top', '10px')
      .on('click', cycleConnectionDensity);
      
    if (viewMode === '3d') {
      // Add rotation controls
      const rotationControls = controls.append('div')
        .attr('class', 'rotation-controls')
        .style('margin-top', '10px');
      
      rotationControls.append('div').text('Rotation Controls:');
      
      const rotateX = rotationControls.append('div').style('margin-top', '5px');
      rotateX.append('span').text('X: ');
      rotateX.append('button').text('-').on('click', () => updateRotation('x', -15));
      rotateX.append('button').text('+').on('click', () => updateRotation('x', 15));
      
      const rotateY = rotationControls.append('div').style('margin-top', '5px');
      rotateY.append('span').text('Y: ');
      rotateY.append('button').text('-').on('click', () => updateRotation('y', -15));
      rotateY.append('button').text('+').on('click', () => updateRotation('y', 15));
      
      const rotateZ = rotationControls.append('div').style('margin-top', '5px');
      rotateZ.append('span').text('Z: ');
      rotateZ.append('button').text('-').on('click', () => updateRotation('z', -15));
      rotateZ.append('button').text('+').on('click', () => updateRotation('z', 15));
      
      rotationControls.append('button')
        .style('margin-top', '10px')
        .text('Reset Rotation')
        .on('click', () => setRotation({ x: 0, y: 0, z: 0 }));
    }
    
    // Calculate positions based on view mode
    if (viewMode === '2d') {
      // 2D layout - vertical positioning
      const verticalSpacing = innerHeight / (processedNodes.length + 1);
      
      processedNodes.forEach((node, i) => {
        node.x = innerWidth / 2;
        node.y = margin.top + verticalSpacing * (i + 1);
        node.z = 0;
      });
    } else {
      // 3D layout - arrange in a 3D space
      // Create a spiral shape for more visual interest
      const spiralRadius = Math.min(innerWidth, innerHeight) / 3;
      const angleStep = 2 * Math.PI / processedNodes.length;
      const heightStep = innerHeight / processedNodes.length;
      
      processedNodes.forEach((node, i) => {
        const angle = i * angleStep;
        const radius = spiralRadius * (1 - i / processedNodes.length * 0.5); // Decreasing radius
        
        // Calculate positions in 3D space
        node.x = innerWidth / 2 + radius * Math.cos(angle);
        node.y = innerHeight / 2 + radius * Math.sin(angle);
        node.z = -(innerHeight / 2) + i * heightStep / 2; // Z dimension for 3D effect
      });
    }
    
    // Create links based on 'next' property and connection density
    links.length = 0; // Clear any existing links
    
    // Sequential connections (from 'next' property)
    if (connectionDensity === 'sequential' || connectionDensity === 'layerwise' || connectionDensity === 'full') {
      processedNodes.forEach((node) => {
        if (node.next) {
          const targetNode = processedNodes.find(n => n.name === node.next);
          if (targetNode) {
            links.push({
              source: node,
              target: targetNode,
              value: 1,
              type: 'sequential'
            });
          }
        }
      });
    }
    
    // Layer-wise connections (every node to every node in adjacent layers)
    if (connectionDensity === 'layerwise' || connectionDensity === 'full') {
      // Identify unique types to group by layers
      const nodeTypes = Array.from(new Set(processedNodes.map(node => node.type)));
      
      // Create layer groups
      const layerGroups = {};
      nodeTypes.forEach(type => {
        layerGroups[type] = processedNodes.filter(node => node.type === type);
      });
      
      // Find the layer sequence based on first connections
      let layerSequence = [];
      let currentType = processedNodes[0]?.type; // Start with the first node's type
      
      while (currentType && layerSequence.length < nodeTypes.length) {
        if (!layerSequence.includes(currentType)) {
          layerSequence.push(currentType);
          
          // Find the next layer type based on connections
          const nodesInCurrentLayer = layerGroups[currentType];
          const nextNodes = nodesInCurrentLayer
            .map(node => processedNodes.find(n => n.name === node.next))
            .filter(Boolean);
          
          if (nextNodes.length > 0) {
            const nextType = nextNodes[0].type;
            if (nextType !== currentType && !layerSequence.includes(nextType)) {
              currentType = nextType;
            } else {
              break;
            }
          } else {
            break;
          }
        } else {
          break;
        }
      }
      
      // Fill in any missing layers
      nodeTypes.forEach(type => {
        if (!layerSequence.includes(type)) {
          layerSequence.push(type);
        }
      });
      
      // Connect every node in each layer to every node in the next layer
      for (let i = 0; i < layerSequence.length - 1; i++) {
        const currentLayerNodes = layerGroups[layerSequence[i]];
        const nextLayerNodes = layerGroups[layerSequence[i + 1]];
        
        if (currentLayerNodes && nextLayerNodes) {
          currentLayerNodes.forEach(source => {
            nextLayerNodes.forEach(target => {
              // Check if this connection already exists from sequential links
              const existingLink = links.find(link => 
                link.source.id === source.id && link.target.id === target.id);
              
              if (!existingLink) {
                links.push({
                  source: source,
                  target: target,
                  value: 0.7,
                  type: 'layerwise'
                });
              }
            });
          });
        }
      }
    }
    
    // Full connections (every node to every other node)
    if (connectionDensity === 'full') {
      processedNodes.forEach((source, i) => {
        processedNodes.forEach((target, j) => {
          if (i !== j) {
            // Check if this connection already exists from sequential or layerwise links
            const existingLink = links.find(link => 
              link.source.id === source.id && link.target.id === target.id);
            
            if (!existingLink) {
              links.push({
                source: source,
                target: target,
                value: 0.3,
                type: 'full',
                isExtra: true
              });
            }
          }
        });
      });
    }
    
    // Additional connections for 3D visualization
    if (viewMode === '3d' && connectionDensity !== 'full') {
      // Only add these extra 3D connections if we're not already in 'full' mode
      processedNodes.forEach((source, i) => {
        // Connect to nodes that are 2 steps away in the sequence
        if (i + 2 < processedNodes.length) {
          links.push({
            source: source,
            target: processedNodes[i + 2],
            value: 0.5, // Thinner connection
            isExtra: true
          });
        }
        
        // Connect to a few random nodes to create more of a network
        if (processedNodes.length > 5 && i % 3 === 0) {
          const randomIdx = (i + Math.floor(Math.random() * processedNodes.length / 2)) % processedNodes.length;
          if (randomIdx !== i && randomIdx !== i + 1) {
            links.push({
              source: source,
              target: processedNodes[randomIdx],
              value: 0.3, // Very thin connection
              isExtra: true
            });
          }
        }
      });
    }
    
    // Apply 3D transformations if in 3D mode
    if (viewMode === '3d') {
      // Function to project 3D coordinates to 2D with rotation
      const project = (node) => {
        // Convert to radians
        const radX = rotation.x * Math.PI / 180;
        const radY = rotation.y * Math.PI / 180;
        const radZ = rotation.z * Math.PI / 180;
        
        // Original coordinates
        let x = node.x - innerWidth / 2;
        let y = node.y - innerHeight / 2;
        let z = node.z;
        
        // Apply X rotation
        let tempY = y;
        let tempZ = z;
        y = tempY * Math.cos(radX) - tempZ * Math.sin(radX);
        z = tempY * Math.sin(radX) + tempZ * Math.cos(radX);
        
        // Apply Y rotation
        let tempX = x;
        tempZ = z;
        x = tempX * Math.cos(radY) + tempZ * Math.sin(radY);
        z = -tempX * Math.sin(radY) + tempZ * Math.cos(radY);
        
        // Apply Z rotation
        tempX = x;
        tempY = y;
        x = tempX * Math.cos(radZ) - tempY * Math.sin(radZ);
        y = tempX * Math.sin(radZ) + tempY * Math.cos(radZ);
        
        // Apply perspective (simple perspective division)
        const focalLength = 800;
        const scale = focalLength / (focalLength + z);
        
        return {
          x: x * scale + innerWidth / 2,
          y: y * scale + innerHeight / 2,
          scale: scale
        };
      };
      
      // Project all nodes
      processedNodes.forEach(node => {
        const projected = project(node);
        node.projectedX = projected.x;
        node.projectedY = projected.y;
        node.projectedScale = projected.scale;
      });
    }
    
    // Add glow filter
    const glowFilter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');
      
    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', '2')
      .attr('result', 'coloredBlur');
      
    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode')
      .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
      .attr('in', 'SourceGraphic');

    // Create arrowhead marker
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#60a5fa');

    // Draw connections first (so they're behind nodes)
    const linksGroup = svg.append('g').attr('class', 'links');
    
    links.forEach((link, i) => {
      // Get positions based on view mode
      const sourceX = viewMode === '3d' ? link.source.projectedX : link.source.x;
      const sourceY = viewMode === '3d' ? link.source.projectedY : link.source.y;
      const targetX = viewMode === '3d' ? link.target.projectedX : link.target.x;
      const targetY = viewMode === '3d' ? link.target.projectedY : link.target.y;
      
      // Create path
      let path;
      if (viewMode === '2d') {
        // Simple curved path for 2D, adjust curvature based on link type
        const curveScale = link.type === 'sequential' ? 1 : 
                         link.type === 'layerwise' ? 0.6 : 0.3;
        const controlX1 = sourceX + (targetX - sourceX) * 0.25 * curveScale;
        const controlY1 = sourceY;
        const controlX2 = sourceX + (targetX - sourceX) * 0.75 * curveScale;
        const controlY2 = targetY;
        path = `M${sourceX},${sourceY} C${controlX1},${controlY1} ${controlX2},${controlY2} ${targetX},${targetY}`;
      } else {
        // For 3D, use straight lines with color gradient for depth perception
        path = `M${sourceX},${sourceY} L${targetX},${targetY}`;
      }
      
      // Determine link color and opacity based on type and view mode
      let color, opacity;
      
      switch(link.type) {
        case 'sequential':
          color = '#60a5fa'; // Primary blue
          opacity = 0.8;
          break;
        case 'layerwise':
          color = '#34d399'; // Green
          opacity = 0.6;
          break;
        case 'full':
          color = '#a78bfa'; // Purple
          opacity = 0.2;
          break;
        default:
          color = viewMode === '3d' && link.isExtra ? '#a78bfa' : '#60a5fa';
          opacity = viewMode === '3d' && link.isExtra ? 0.3 : 0.6;
      }
      
      // Calculate z-order based on z-position for 3D mode
      const zIndex = viewMode === '3d' ? 
                    Math.round(1000 - (link.source.z + link.target.z) / 2) : 0;
      
      // Determine link width based on parameters and z-position for 3D
      let strokeWidth = link.type === 'sequential' ? 2 : 
                        link.type === 'layerwise' ? 1.5 : 1;
                        
      if (viewMode === '3d') {
        // Adjust width by scale for perspective effect
        const avgScale = (link.source.projectedScale + link.target.projectedScale) / 2;
        strokeWidth = link.value * 2 * avgScale;
      }
      
      linksGroup.append('path')
        .attr('d', path)
        .attr('stroke', color)
        .attr('stroke-width', strokeWidth)
        .attr('fill', 'none')
        .attr('opacity', opacity)
        .attr('marker-end', link.type === 'sequential' ? 'url(#arrowhead)' : null)
        .attr('data-z-index', zIndex)
        .attr('data-link-type', link.type)
        .classed('extra-link', link.isExtra);
    });

    // Draw nodes
    const nodesGroup = svg.append('g').attr('class', 'nodes');
    
    // Sort nodes by z-index for 3D mode to ensure proper layering
    if (viewMode === '3d') {
      processedNodes.sort((a, b) => b.z - a.z);
    }
    
    const nodeElements = nodesGroup.selectAll('.node')
      .data(processedNodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => {
        if (viewMode === '3d') {
          return `translate(${d.projectedX}, ${d.projectedY})`;
        }
        return `translate(${d.x}, ${d.y})`;
      })
      .attr('data-z-index', d => viewMode === '3d' ? Math.round(1000 - d.z) : 0);
    
    // Function to determine node size based on parameters
    const getNodeSize = (params, scale = 1) => {
      if (params === 0) return 10 * scale;
      return Math.max(10, Math.min(25, 10 + Math.log(params) / 2)) * scale;
    };
    
    // Function to determine node color based on type
    function getNodeColor(type) {
      const colors = {
        'input': '#60a5fa', // blue
        'conv1': '#4ade80', // green
        'conv2': '#4ade80', // green
        'pool1': '#a78bfa', // purple
        'pool2': '#a78bfa', // purple
        'fc1': '#f97316', // orange
        'fc2': '#f97316', // orange
        'relu1': '#ec4899', // pink
        'relu2': '#ec4899', // pink
        'relu3': '#ec4899', // pink
        'output': '#f43f5e' // red
      };
      
      // Check if the type is in our color map (case insensitive)
      const lowerType = type.toLowerCase();
      for (const key in colors) {
        if (lowerType.includes(key)) {
          return colors[key];
        }
      }
      
      return '#9ca3af'; // gray default
    }
    
    // Add circles for nodes
    nodeElements.append('circle')
      .attr('r', d => {
        if (viewMode === '3d') {
          return getNodeSize(d.parameters, d.projectedScale);
        }
        return getNodeSize(d.parameters);
      })
      .attr('fill', d => getNodeColor(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', d => viewMode === '3d' ? 1.5 * d.projectedScale : 1.5)
      .attr('filter', 'url(#glow)')
      .attr('class', 'node-circle')
      .on('mouseover', function(event, d) {
        // Highlight on hover
        d3.select(this)
          .transition()
          .duration(300)
          .attr('r', viewMode === '3d' ? 
                getNodeSize(d.parameters, d.projectedScale) * 1.3 : 
                getNodeSize(d.parameters) * 1.3);
          
        // Show tooltip
        tooltip
          .style('opacity', 1)
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 28}px`)
          .html(`
            <div class="tooltip-title">${d.type}</div>
            <div class="tooltip-content">
              <strong>Name:</strong> ${d.name}<br/>
              <strong>Parameters:</strong> ${d.parameters.toLocaleString()}<br/>
              ${d.next ? `<strong>Next:</strong> ${d.next}<br/>` : ''}
              ${structure[d.id].module_path ? `<strong>Path:</strong> ${JSON.stringify(structure[d.id].module_path)}<br/>` : ''}
              ${structure[d.id].shape ? `<strong>Shape:</strong> ${JSON.stringify(structure[d.id].shape)}<br/>` : ''}
              ${viewMode === '3d' ? `<strong>3D Position:</strong> (${Math.round(d.x)}, ${Math.round(d.y)}, ${Math.round(d.z)})<br/>` : ''}
            </div>
          `);
      })
      .on('mouseout', function(event, d) {
        // Restore on mouseout
        d3.select(this)
          .transition()
          .duration(300)
          .attr('r', viewMode === '3d' ? 
                getNodeSize(d.parameters, d.projectedScale) : 
                getNodeSize(d.parameters));
          
        // Hide tooltip
        tooltip.style('opacity', 0);
      });
    
    // Add labels - adjust for 3D mode
    nodeElements.append('text')
      .attr('dy', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', d => viewMode === '3d' ? `${Math.max(8, 12 * d.projectedScale)}px` : '12px')
      .attr('font-weight', 'bold')
      .text(d => d.name);
      
    nodeElements.append('text')
      .attr('dy', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e5e7eb')
      .attr('font-size', d => viewMode === '3d' ? `${Math.max(6, 10 * d.projectedScale)}px` : '10px')
      .text(d => d.parameters > 0 ? `${d.parameters.toLocaleString()} params` : '');

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text(`Neural Network Architecture (${viewMode.toUpperCase()})`);
    
    // Add legend
    const legendData = [
      {type: 'input', label: 'Input', color: getNodeColor('input')},
      {type: 'conv1', label: 'Convolution', color: getNodeColor('conv1')},
      {type: 'fc1', label: 'Fully Connected', color: getNodeColor('fc1')},
      {type: 'pool1', label: 'Pooling', color: getNodeColor('pool1')},
      {type: 'relu1', label: 'ReLU', color: getNodeColor('relu1')},
      {type: 'sequential', label: 'Sequential Connection', color: '#60a5fa', isLink: true},
      {type: 'layerwise', label: 'Layer Connection', color: '#34d399', isLink: true, showIf: ['layerwise', 'full']},
      {type: 'full', label: 'All Connections', color: '#a78bfa', isLink: true, showIf: ['full']}
    ].filter(item => !item.showIf || item.showIf.includes(connectionDensity));
    
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 180}, 20)`);
      
    const legendItems = legend.selectAll('.legend-item')
      .data(legendData)
      .enter()
      .append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 25})`);
      
    // Add symbol (circle for nodes, line for connections)
    legendItems.each(function(d) {
      const item = d3.select(this);
      if (d.isLink) {
        // Line for connections
        item.append('line')
          .attr('x1', 0)
          .attr('y1', 5)
          .attr('x2', 20)
          .attr('y2', 5)
          .attr('stroke', d.color)
          .attr('stroke-width', 2)
          .attr('opacity', d.type === 'full' ? 0.3 : 0.8);
      } else {
        // Circle for nodes
        item.append('circle')
          .attr('r', 6)
          .attr('cx', 10)
          .attr('cy', 5)
          .attr('fill', d.color)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);
      }
    });
      
    legendItems.append('text')
      .attr('x', 25)
      .attr('y', 8)
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .text(d => d.label);
    
    // Add extra info about 3D mode
    if (viewMode === '3d') {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height - 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e5e7eb')
        .attr('font-size', '12px')
        .text('Use rotation controls to explore the 3D visualization');
    }
    
    // Cleanup function
    return () => {
      if (svgRef.current && svgRef.current.parentNode) {
        svgRef.current.parentNode.removeChild(svgRef.current);
      }
    };
  }, [structure, viewMode, rotation, connectionDensity]);

  // Function to update rotation
  const updateRotation = (axis, angle) => {
    setRotation(prev => ({
      ...prev,
      [axis]: prev[axis] + angle
    }));
  };

  return (
    <div 
      className={`network-visualization-container ${viewMode === '3d' ? 'mode-3d' : ''}`} 
      ref={containerRef}
    >
      {(!structure || !structure.length) && (
        <div className="no-data-message">No model structure available</div>
      )}
    </div>
  );
};

export default NetworkVisualization; 
