import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './ModelGraph.css';

const ModelGraph = ({ structure }) => {
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

    // Define arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('xoverflow', 'visible')
      .append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999')
      .style('stroke', 'none');
    
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

    // Create nodes and links from sequential structure
    const nodes = structure;
    const links = [];
    
    // Create links based on 'next' property
    for (let i = 0; i < nodes.length - 1; i++) {
      if (nodes[i].next) {
        links.push({
          source: nodes[i],
          target: nodes.find(n => n.name === nodes[i].next)
        });
      }
    }

    // Group nodes by module
    const moduleGroups = {};
    nodes.forEach(node => {
      if (!moduleGroups[node.module_name]) {
        moduleGroups[node.module_name] = [];
      }
      moduleGroups[node.module_name].push(node);
    });

    // Calculate module positions and filter out single-node modules
    const moduleOrder = Object.keys(moduleGroups).filter(
      moduleName => moduleGroups[moduleName].length > 1
    );
    const modulePositions = {};
    moduleOrder.forEach((moduleName, index) => {
      modulePositions[moduleName] = {
        x: (width - margin.left - margin.right) * (index / (moduleOrder.length - 1)),
        y: height / 2
      };
    });

    // Create force simulation with modified forces for clustered layout
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links)
        .id(d => d.name)
        .distance(50))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('x', d3.forceX(d => {
        // For nodes in multi-node modules, use module position
        // For single nodes, space them evenly
        const moduleGroup = moduleGroups[d.module_name];
        if (moduleGroup.length > 1) {
          return modulePositions[d.module_name].x;
        } else {
          return (width - margin.left - margin.right) * (Array.from(nodes).indexOf(d) / (nodes.length - 1));
        }
      }).strength(0.5))
      .force('y', d3.forceY(d => modulePositions[d.module_name]?.y || height / 2).strength(0.1))
      .force('collision', d3.forceCollide().radius(30));

    // Add module background rectangles only for modules with multiple nodes
    const moduleBackgrounds = g.append('g')
      .selectAll('.module-background')
      .data(moduleOrder)
      .join('rect')
      .attr('class', 'module-background')
      .attr('rx', 10)
      .attr('ry', 10)
      .attr('fill', '#f5f5f5')
      .attr('stroke', '#ddd')
      .attr('stroke-width', 1)
      .attr('opacity', 0.5);

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
        'Output': '#F44336',
        'ReLU': '#FF5722',
        'Activation': '#FF5722'
      };
      return colors[type] || '#9E9E9E';
    };

    // Create links with arrows
    const link = g.append('g')
      .selectAll('path')
      .data(links)
      .join('path')
      .attr('class', 'link')
      .attr('marker-end', 'url(#arrowhead)');

    // Create node groups
    const node = g.append('g')
      .selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add node circles
    node.append('circle')
      .attr('r', 8)
      .attr('class', 'node-circle')
      .style('fill', d => getNodeColor(d.type));

    // Add node labels
    node.append('text')
      .attr('dy', '0.31em')
      .attr('x', 12)
      .attr('text-anchor', 'start')
      .text(d => d.name.split('.').pop()) // Show only the last part of the name
      .clone(true)
      .lower()
      .attr('stroke', 'white')
      .attr('stroke-width', 3);

    // Add module labels
    const moduleLabels = g.append('g')
      .selectAll('.module-label')
      .data(moduleOrder)
      .join('text')
      .attr('class', 'module-label')
      .attr('text-anchor', 'middle')
      .style('font-weight', 'bold')
      .style('font-size', '14px')
      .text(d => d);

    // Add tooltips
    const tooltip = container
      .append('div')
      .attr('class', 'model-graph-tooltip')
      .style('opacity', 0);

    node
      .on('mouseover', (event, d) => {
        tooltip
          .style('opacity', 1)
          .html(`
            <div class="tooltip-title">${d.type}</div>
            <div class="tooltip-content">
              <strong>Name:</strong> ${d.name}<br/>
              <strong>Module:</strong> ${d.module_name}<br/>
              ${d.shape ? `<strong>Shape:</strong> ${JSON.stringify(d.shape)}<br/>` : ''}
              ${d.parameters ? `<strong>Parameters:</strong> ${d.parameters.toLocaleString()}<br/>` : ''}
              ${d.trainable ? `<strong>Trainable:</strong> ${d.trainable.toLocaleString()}` : ''}
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
      'ReLU': '#FF5722',
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

    // Simulation tick function
    simulation.on('tick', () => {
      // Update module backgrounds only for modules with multiple nodes
      moduleBackgrounds
        .attr('x', d => {
          const nodes = moduleGroups[d];
          if (nodes.length <= 1) return 0;
          const x = d3.min(nodes, n => n.x) - 30;
          return x;
        })
        .attr('y', d => {
          const nodes = moduleGroups[d];
          if (nodes.length <= 1) return 0;
          const y = d3.min(nodes, n => n.y) - 30;
          return y;
        })
        .attr('width', d => {
          const nodes = moduleGroups[d];
          if (nodes.length <= 1) return 0;
          const minX = d3.min(nodes, n => n.x);
          const maxX = d3.max(nodes, n => n.x);
          return maxX - minX + 60;
        })
        .attr('height', d => {
          const nodes = moduleGroups[d];
          if (nodes.length <= 1) return 0;
          const minY = d3.min(nodes, n => n.y);
          const maxY = d3.max(nodes, n => n.y);
          return maxY - minY + 60;
        });

      // Update module labels only for modules with multiple nodes
      moduleLabels
        .attr('x', d => {
          const nodes = moduleGroups[d];
          if (nodes.length <= 1) return 0;
          const minX = d3.min(nodes, n => n.x);
          const maxX = d3.max(nodes, n => n.x);
          return (minX + maxX) / 2;
        })
        .attr('y', d => {
          const nodes = moduleGroups[d];
          if (nodes.length <= 1) return 0;
          const minY = d3.min(nodes, n => n.y) - 10;
          return minY;
        })
        .style('opacity', d => moduleGroups[d].length > 1 ? 1 : 0);

      // Update links
      link.attr('d', d => {
        const dx = d.target.x - d.source.x;
        const dy = d.target.y - d.source.y;
        return `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`;
      });

      // Update nodes
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
    
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
