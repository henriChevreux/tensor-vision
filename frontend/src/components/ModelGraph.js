import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const ModelGraph = ({ structure }) => {
  const svgRef = useRef();
  
  useEffect(() => {
    if (!structure || !structure.length) return;
    
    // Very basic rendering of model structure
    const width = 800;
    const height = 600;
    
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
      
    svg.selectAll('*').remove();
    
    // Create hierarchical layout
    const root = d3.hierarchy(structure[0]);
    
    const treeLayout = d3.tree()
      .size([width - 100, height - 100]);
      
    treeLayout(root);
    
    // Draw links
    svg.append('g')
      .selectAll('line')
      .data(root.links())
      .enter()
      .append('line')
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
      .attr('stroke', '#ccc')
      .attr('stroke-width', 1);
      
    // Draw nodes
    const nodes = svg.append('g')
      .selectAll('g')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x}, ${d.y})`);
      
    nodes.append('circle')
      .attr('r', 5)
      .attr('fill', '#69b3a2');
      
    nodes.append('text')
      .attr('dy', '.31em')
      .attr('x', d => d.children ? -8 : 8)
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .text(d => `${d.data.type} (${d.data.name})`);
      
  }, [structure]);
  
  return (
    <div>
      <h3>Model Structure</h3>
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default ModelGraph;
