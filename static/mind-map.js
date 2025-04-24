/**
 * Mind Map Visualization with Gravitational Physics
 * Creates an interactive mind map where nodes represent ideas with parent-child relationships
 * Nodes orbit around their parent nodes using a gravitational physics system
 */

class MindMap {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container element with ID '${containerId}' not found.`);
      return;
    }

    // Default options with Fred's color scheme
    this.options = {
      centralTopic: options.centralTopic || "Research Mind Maps",
      nodeSize: options.nodeSize || { min: 6, max: 30 }, // Size range based on child count
      colors: options.colors || {
        node: 'var(--fred-book-cloth)', // Default/fallback
        // childNode: 'var(--fred-kraft)', // Keep or remove if unused
        connection: 'rgba(var(--fred-book-cloth-rgb), 0.3)',
        text: 'var(--text-primary)',
        centralNode: 'var(--fred-focus)', // Maybe use semantic/episodic color?
        brainPulse: 'rgba(var(--fred-focus-rgb), 0.6)',
        semanticNode: 'var(--fred-blue)', // New color for Semantic
        episodicNode: 'var(--fred-green)' // New color for Episodic
      },
      orbitSpeed: options.orbitSpeed || 0.005, // Very slow for subtle movement
      distanceFactor: options.distanceFactor || 120, // Distance between parent and child nodes
      gravitationalConstant: options.gravitationalConstant || 0.02, // Controls orbit tightness
      repulsionForce: options.repulsionForce || 10, // Keeps nodes from getting too close
      active: false,
      graphApiUrl: options.graphApiUrl || '/graph', // API endpoint
      initialCenterNodeId: options.initialCenterNodeId || null, // Optional specific starting node
      initialDepth: options.initialDepth || 1, // Default depth
      ...options
    };

    this.nodes = [];
    this.connections = [];
    this.isInitialized = false;
    this.animationFrame = null;
    this.pulsePhase = 0; // For brain pulsing animation

    // Initialize the SVG after a small delay to ensure container is fully rendered
    setTimeout(() => this.initSvg(), 10);
  }

  initSvg() {
    // Clear existing content
    this.container.innerHTML = '';

    // Ensure the container takes up the full viewport
    this.container.style.position = 'fixed';
    this.container.style.top = '0';
    this.container.style.left = '0';
    this.container.style.width = '100vw';
    this.container.style.height = '100vh';
    this.container.style.overflow = 'hidden';
    this.container.style.pointerEvents = 'none'; // Allow clicks to pass through to UI elements
    
    // Get window dimensions instead of container to ensure full coverage
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    
    // Create SVG element
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.setAttribute('width', '100%');
    this.svg.setAttribute('height', '100%');
    this.svg.setAttribute('viewBox', `0 0 ${this.width} ${this.height}`);
    this.svg.style.position = 'absolute';
    this.svg.style.top = '0';
    this.svg.style.left = '0';
    this.svg.style.width = '100%';
    this.svg.style.height = '100%';
    this.svg.style.pointerEvents = 'auto';
    this.svg.style.zIndex = '-1'; // Ensure it stays in the background
    
    // Create glow filters for nodes
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    
    // Standard glow filter
    const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
    filter.setAttribute('id', 'node-glow');
    filter.setAttribute('x', '-50%');
    filter.setAttribute('y', '-50%');
    filter.setAttribute('width', '200%');
    filter.setAttribute('height', '200%');
    
    const feGaussianBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
    feGaussianBlur.setAttribute('stdDeviation', '2');
    feGaussianBlur.setAttribute('result', 'coloredBlur');
    
    const feMerge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
    const feMergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
    feMergeNode1.setAttribute('in', 'coloredBlur');
    const feMergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
    feMergeNode2.setAttribute('in', 'SourceGraphic');
    
    feMerge.appendChild(feMergeNode1);
    feMerge.appendChild(feMergeNode2);
    filter.appendChild(feGaussianBlur);
    filter.appendChild(feMerge);
    defs.appendChild(filter);
    
    // Central brain node glow filter (stronger)
    const centralFilter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
    centralFilter.setAttribute('id', 'central-node-glow');
    centralFilter.setAttribute('x', '-50%');
    centralFilter.setAttribute('y', '-50%');
    centralFilter.setAttribute('width', '200%');
    centralFilter.setAttribute('height', '200%');
    
    const centralBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
    centralBlur.setAttribute('stdDeviation', '5');
    centralBlur.setAttribute('result', 'centralBlur');
    
    const centralMerge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
    const centralMergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
    centralMergeNode1.setAttribute('in', 'centralBlur');
    const centralMergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
    centralMergeNode2.setAttribute('in', 'SourceGraphic');
    
    centralMerge.appendChild(centralMergeNode1);
    centralMerge.appendChild(centralMergeNode2);
    centralFilter.appendChild(centralBlur);
    centralFilter.appendChild(centralMerge);
    defs.appendChild(centralFilter);
    
    // Add brain pulse animation
    const brainPulseAnimation = document.createElementNS('http://www.w3.org/2000/svg', 'radialGradient');
    brainPulseAnimation.setAttribute('id', 'brainPulseGradient');
    brainPulseAnimation.setAttribute('cx', '50%');
    brainPulseAnimation.setAttribute('cy', '50%');
    brainPulseAnimation.setAttribute('r', '50%');
    brainPulseAnimation.setAttribute('fx', '50%');
    brainPulseAnimation.setAttribute('fy', '50%');
    
    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', this.options.colors.centralNode);
    
    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '70%');
    stop2.setAttribute('stop-color', this.options.colors.centralNode);
    
    const stop3 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop3.setAttribute('offset', '100%');
    stop3.setAttribute('stop-color', this.options.colors.brainPulse);
    
    brainPulseAnimation.appendChild(stop1);
    brainPulseAnimation.appendChild(stop2);
    brainPulseAnimation.appendChild(stop3);
    defs.appendChild(brainPulseAnimation);
    
    // Neural texture pattern for brain node
    const neuralPattern = document.createElementNS('http://www.w3.org/2000/svg', 'pattern');
    neuralPattern.setAttribute('id', 'neuralPattern');
    neuralPattern.setAttribute('width', '40');
    neuralPattern.setAttribute('height', '40');
    neuralPattern.setAttribute('patternUnits', 'userSpaceOnUse');
    neuralPattern.setAttribute('patternTransform', 'rotate(30)');
    
    // Add neural texture lines
    for (let i = 0; i < 8; i++) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      const x1 = Math.random() * 40;
      const y1 = Math.random() * 40;
      const x2 = Math.random() * 40;
      const y2 = Math.random() * 40;
      
      line.setAttribute('x1', x1);
      line.setAttribute('y1', y1);
      line.setAttribute('x2', x2);
      line.setAttribute('y2', y2);
      line.setAttribute('stroke', 'rgba(255, 255, 255, 0.4)');
      line.setAttribute('stroke-width', '0.5');
      
      neuralPattern.appendChild(line);
    }
    
    // Add some neural dots
    for (let i = 0; i < 10; i++) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', Math.random() * 40);
      circle.setAttribute('cy', Math.random() * 40);
      circle.setAttribute('r', 0.5 + Math.random() * 1.5);
      circle.setAttribute('fill', 'rgba(255, 255, 255, 0.6)');
      
      neuralPattern.appendChild(circle);
    }
    
    defs.appendChild(neuralPattern);
    
    this.svg.appendChild(defs);
    
    // Create groups for connections and nodes
    this.connectionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    this.nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    
    this.svg.appendChild(this.connectionsGroup);
    this.svg.appendChild(this.nodesGroup);
    
    this.container.appendChild(this.svg);
    
    // Load initial graph data from the API
    this.loadGraphData();

    this.isInitialized = true;
    
    // Start animation
    this.startAnimation();
  }

  // Render all nodes and connections
  renderGraph() {
    // Clear previous elements
    this.connectionsGroup.innerHTML = '';
    this.nodesGroup.innerHTML = '';
    
    // Create neural synapse group
    this.synapseGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    this.svg.appendChild(this.synapseGroup);
    
    // Render connections first (so they appear behind nodes)
    this.connections.forEach(connection => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', connection.source.x);
      line.setAttribute('y1', connection.source.y);
      line.setAttribute('x2', connection.target.x);
      line.setAttribute('y2', connection.target.y);
      line.setAttribute('stroke', this.options.colors.connection);
      line.setAttribute('stroke-width', 1.5 * connection.strength);
      line.setAttribute('opacity', 0.7);
      this.connectionsGroup.appendChild(line);
      connection.element = line;
    });
    
    // Render nodes
    this.nodes.forEach(node => {
      // Create group for node and label
      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.setAttribute('transform', `translate(${node.x}, ${node.y})`);

      // Create circle element
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('r', node.size);
      // --- COLORING LOGIC ---
      let nodeColor = this.options.colors.node; // Default
      if (node.type === 'Semantic') {
          nodeColor = this.options.colors.semanticNode;
      } else if (node.type === 'Episodic') {
          nodeColor = this.options.colors.episodicNode;
      } else if (node.isCentral) { // Keep central node distinct? Or color it too?
          nodeColor = this.options.colors.centralNode;
      }
      // --- END COLORING LOGIC ---

      circle.setAttribute('fill', nodeColor);
      circle.setAttribute('stroke', 'rgba(var(--fred-book-cloth-rgb), 0.5)');
      circle.setAttribute('stroke-width', 1);
      circle.style.cursor = 'pointer';

      // Apply glow filter
      circle.setAttribute('filter', node.isCentral ? 'url(#central-node-glow)' : 'url(#node-glow)');


      // Add brain pulse effect if it's the central node
      if (node.isCentral) {
        // Use radial gradient for pulse
        circle.setAttribute('fill', 'url(#brainPulseGradient)');
        // Optionally add neural texture overlay
        const textureRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        textureRect.setAttribute('x', -node.size);
        textureRect.setAttribute('y', -node.size);
        textureRect.setAttribute('width', node.size * 2);
        textureRect.setAttribute('height', node.size * 2);
        textureRect.setAttribute('fill', 'url(#neuralPattern)');
        textureRect.setAttribute('opacity', '0.3');
        textureRect.style.pointerEvents = 'none'; // Prevent texture from blocking clicks
        group.appendChild(textureRect);
      }

      group.appendChild(circle);

      // Create text label element
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('dy', '.3em'); // Vertically center
      text.setAttribute('text-anchor', 'middle'); // Horizontally center
      text.setAttribute('fill', this.options.colors.text);
      text.setAttribute('font-size', `${Math.max(10, node.size * 0.5)}px`); // Scale font size
      text.textContent = node.label;
      text.style.pointerEvents = 'none'; // Labels don't block clicks on nodes

      group.appendChild(text);

      // Add click listener to nodes
      group.addEventListener('click', () => this.handleNodeClick(node));
      // Add hover listeners for tooltip/info display (optional)
      group.addEventListener('mouseover', (event) => this.showNodeInfo(node, event));
      group.addEventListener('mouseout', () => this.hideNodeInfo());


      this.nodesGroup.appendChild(group);
      node.element = group; // Store reference to the group
    });
    
    // Create synapse elements
    this.synapses.forEach(synapse => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('stroke', 'rgba(var(--fred-focus-rgb), 0.8)');
      line.setAttribute('stroke-width', '2');
      line.setAttribute('opacity', '0');
      this.synapseGroup.appendChild(line);
      synapse.element = line;
    });
  }

  // Update node positions based on physics
  update() {
    // Skip if not initialized or not active
    if (!this.isInitialized) return;
    
    // Update brain pulse animation
    this.updateBrainPulse();
    
    // Update neural synapses
    this.updateSynapses();
    
    // Apply gravitational and repulsion forces
    this.applyForces();
    
    // Update orbital positions
    this.updateOrbits();
    
    // Update node positions and render
    this.updateNodePositions();
  }
  
  // Update brain pulse animation
  updateBrainPulse() {
    // Get central node
    const brainNode = this.nodes.find(node => node.isCentral);
    if (!brainNode || !brainNode.pulseRing) return;
    
    // Increment pulse phase
    this.pulsePhase += 0.02;
    
    // Calculate pulse size (oscillating)
    const pulseSize = brainNode.size + 5 + Math.sin(this.pulsePhase) * 8;
    
    // Update pulse ring
    brainNode.pulseRing.setAttribute('r', pulseSize);
    brainNode.pulseRing.setAttribute('opacity', 0.3 + Math.sin(this.pulsePhase) * 0.2);
    
    // Occasionally trigger synapses when actively thinking
    if (this.options.active && Math.random() > 0.95) {
      const randomSynapse = this.synapses[Math.floor(Math.random() * this.synapses.length)];
      if (!randomSynapse.active) {
        randomSynapse.active = true;
        randomSynapse.progress = 0;
        
        // Pick a random node to connect to
        const randomNode = this.nodes[Math.floor(Math.random() * this.nodes.length)];
        if (randomNode && randomNode !== brainNode) {
          randomSynapse.startX = randomNode.x;
          randomSynapse.startY = randomNode.y;
          randomSynapse.endX = brainNode.x;
          randomSynapse.endY = brainNode.y;
        }
      }
    }
  }
  
  // Update neural synapses
  updateSynapses() {
    if (!this.synapses) return;
    
    const brainNode = this.nodes.find(node => node.isCentral);
    if (!brainNode) return;
    
    this.synapses.forEach(synapse => {
      if (synapse.active) {
        // Update progress
        synapse.progress += synapse.speed;
        
        // Calculate current position along the path
        const x = synapse.startX + (synapse.endX - synapse.startX) * synapse.progress;
        const y = synapse.startY + (synapse.endY - synapse.startY) * synapse.progress;
        
        // Update line position
        synapse.element.setAttribute('x1', synapse.startX);
        synapse.element.setAttribute('y1', synapse.startY);
        synapse.element.setAttribute('x2', x);
        synapse.element.setAttribute('y2', y);
        
        // Fade in/out based on progress
        let opacity = 0;
        if (synapse.progress < 0.2) {
          opacity = synapse.progress * 5; // Fade in
        } else if (synapse.progress > 0.8) {
          opacity = (1 - synapse.progress) * 5; // Fade out
        } else {
          opacity = 1; // Full opacity in the middle
        }
        
        synapse.element.setAttribute('opacity', opacity);
        
        // Reset when complete
        if (synapse.progress >= 1) {
          synapse.active = false;
          synapse.cooldown = 30 + Math.floor(Math.random() * 100);
        }
      } else {
        // Decrement cooldown
        if (synapse.cooldown > 0) {
          synapse.cooldown--;
        }
        
        // Maybe activate if cooled down and in active mode
        if (synapse.cooldown === 0 && this.options.active && Math.random() > 0.99) {
          synapse.active = true;
          synapse.progress = 0;
          
          // Choose a random node as the starting point
          const randomNode = this.nodes[Math.floor(Math.random() * this.nodes.length)];
          if (randomNode && randomNode !== brainNode) {
            synapse.startX = randomNode.x;
            synapse.startY = randomNode.y;
            synapse.endX = brainNode.x;
            synapse.endY = brainNode.y;
          }
        }
      }
    });
  }
  
  // Apply gravitational and repulsion forces
  applyForces() {
    // Reset forces
    this.nodes.forEach(node => {
      node.vx = 0;
      node.vy = 0;
    });
    
    // Apply repulsion between all nodes to prevent overlaps
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const nodeA = this.nodes[i];
        const nodeB = this.nodes[j];
        
        // Skip repulsion calculation if they're parent-child (handled by orbits)
        if (nodeA.parent === nodeB || nodeB.parent === nodeA) continue;
        
        const dx = nodeB.x - nodeA.x;
        const dy = nodeB.y - nodeA.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const minDistance = nodeA.size + nodeB.size + 20; // Minimum desired distance
        
        if (distance < minDistance) {
          const force = this.options.repulsionForce / Math.max(1, distance);
          const forceX = (dx / distance) * force;
          const forceY = (dy / distance) * force;
          
          if (!nodeA.fixed) {
            nodeA.vx -= forceX;
            nodeA.vy -= forceY;
          }
          
          if (!nodeB.fixed) {
            nodeB.vx += forceX;
            nodeB.vy += forceY;
          }
        }
      }
    }
  }
  
  // Update orbital positions of nodes
  updateOrbits() {
    this.nodes.forEach(node => {
      // Skip central/fixed nodes
      if (node.fixed || !node.parent) return;
      
      // Update orbit angle
      node.orbitAngle += node.orbitSpeed;
      
      // Calculate new position based on orbit
      node.x = node.parent.x + Math.cos(node.orbitAngle) * node.orbitRadius;
      node.y = node.parent.y + Math.sin(node.orbitAngle) * node.orbitRadius;
    });
  }
  
  // Update node positions in the DOM
  updateNodePositions() {
    // Update connections first
    this.connections.forEach(connection => {
      connection.element.setAttribute('x1', connection.source.x);
      connection.element.setAttribute('y1', connection.source.y);
      connection.element.setAttribute('x2', connection.target.x);
      connection.element.setAttribute('y2', connection.target.y);
    });
    
    // Update nodes
    this.nodes.forEach(node => {
      // Apply velocity
      if (!node.fixed) {
        node.x += node.vx;
        node.y += node.vy;
      }
      
      // Update node and label positions
      if (node.isCentral) {
        // Update brain node components
        node.brainCircle.setAttribute('cx', node.x);
        node.brainCircle.setAttribute('cy', node.y);
        node.pulseRing.setAttribute('cx', node.x);
        node.pulseRing.setAttribute('cy', node.y);
        // Update all other circles in the group
        const circles = node.group.querySelectorAll('circle');
        circles.forEach(circle => {
          if (circle !== node.brainCircle && circle !== node.pulseRing) {
            circle.setAttribute('cx', node.x);
            circle.setAttribute('cy', node.y);
          }
        });
      } else {
        node.element.setAttribute('cx', node.x);
        node.element.setAttribute('cy', node.y);
      }
      
      node.labelElement.setAttribute('x', node.x);
      node.labelElement.setAttribute('y', node.y + node.size + 10);
    });
  }

  // Start animation loop
  startAnimation() {
    const animate = () => {
      this.update();
      this.animationFrame = requestAnimationFrame(animate);
    };
    
    animate();
  }

  // Set active state - activates brain thinking mode
  setActive(active) {
    this.options.active = active;
    
    // Get brain node
    const brainNode = this.nodes.find(node => node.isCentral);
    if (!brainNode || !brainNode.brainCircle) return;
    
    if (active) {
      // Make brain more active - increase pulsing and neural activity
      brainNode.brainCircle.setAttribute('fill', 'url(#brainPulseGradient)');
      brainNode.pulseRing.setAttribute('stroke-width', '3');
    } else {
      // Calm down brain activity
      brainNode.pulseRing.setAttribute('stroke-width', '2');
    }
  }

  // Resize handler - updated to properly recenter
  resize() {
    if (!this.isInitialized) return;
    
    // Use window dimensions for full coverage
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    
    this.svg.setAttribute('viewBox', `0 0 ${this.width} ${this.height}`);
    
    // Reposition central node to window center
    const centralNode = this.nodes.find(node => node.isCentral);
    if (centralNode) {
      centralNode.x = this.width / 2;
      centralNode.y = this.height / 2;
    }
    
    // Re-render the graph
    this.renderGraph();
  }

  // Clean up
  destroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    
    if (this.container) {
      this.container.innerHTML = '';
    }
  }

  // Load data from hybrid memory system
  loadMemoryData(dataFile = null) {
    const filePath = dataFile || this.options.dataFile || 'static/memory_map_data.json';
    
    fetch(filePath)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load memory data: ${response.status} ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        this.initFromMemoryData(data);
      })
      .catch(error => {
        console.error('Error loading memory data:', error);
      });
  }
  
  // Initialize mind map from memory data
  initFromMemoryData(data) {
    // Clear existing nodes and connections
    this.nodes = [];
    this.connections = [];
    if (this.nodesGroup) this.nodesGroup.innerHTML = '';
    if (this.connectionsGroup) this.connectionsGroup.innerHTML = '';

    if (!data || !data.nodes || !data.connections) {
      console.error('Invalid memory data format');
      // Initialize with just a central node if data is bad
      this.nodes = [{
        id: 'central_fallback',
        label: 'AI Brain',
        x: this.width / 2,
        y: this.height / 2,
        size: this.options.nodeSize.max,
        color: this.options.colors.centralNode,
        isCentral: true,
        fixed: true,
        parent: null,
        orbitAngle: 0,
        orbitRadius: 0,
        orbitSpeed: 0,
        vx: 0,
        vy: 0,
        connectionCount: 0,
        totalStrength: 0
      }];
      this.connections = [];
      this.synapses = [];
      this.renderGraph(); // Render the fallback central node
      return;
    }

    const memoryNodes = data.nodes;
    const memoryConnections = data.connections;
    const centerX = this.width / 2;
    const centerY = this.height / 2;

    let centralNode = null;
    const nodeMap = new Map(); // Map index to node object
    const nodeConnections = new Map(); // Map node object to its connections

    // First pass: Create all nodes and find the central one
    memoryNodes.forEach((memNode, index) => {
      const isCentral = memNode.isCentral || false; // Explicitly check isCentral

      const node = {
        id: memNode.id,
        label: memNode.label || `Node ${index}`,
        x: centerX, // Initial position, will be updated
        y: centerY,
        size: this.options.nodeSize.min, // Start with min size
        color: memNode.color || (isCentral ? this.options.colors.centralNode : this.options.colors.childNode),
        isCentral: isCentral,
        vx: 0,
        vy: 0,
        fixed: isCentral,
        parent: null, // Will be determined later
        orbitAngle: Math.random() * 2 * Math.PI, // Random initial angle
        orbitRadius: 0, // Will be set later
        orbitSpeed: 0, // Will be set later
        connectionCount: 0, // Track connections
        totalStrength: 0, // Track total strength
        index: index // Store original index for connection lookup
      };

      if (isCentral) {
        node.x = centerX;
        node.y = centerY;
        node.size = this.options.nodeSize.max;
        centralNode = node;
      }

      this.nodes.push(node);
      nodeMap.set(index, node);
      nodeConnections.set(node, []); // Initialize connection list for each node
    });

    // Ensure a central node exists
    if (!centralNode && this.nodes.length > 0) {
      centralNode = this.nodes[0];
      centralNode.isCentral = true;
      centralNode.fixed = true;
      centralNode.x = centerX;
      centralNode.y = centerY;
      centralNode.color = this.options.colors.centralNode;
      centralNode.size = this.options.nodeSize.max;
    } else if (!centralNode && this.nodes.length === 0) {
        console.error("No nodes found in data and failed to create fallback.");
        // Handle the case where no nodes could be established
         this.nodes = [{
            id: 'central_fallback_empty', label: 'AI Brain', x: this.width / 2, y: this.height / 2,
            size: this.options.nodeSize.max, color: this.options.colors.centralNode, isCentral: true, fixed: true,
            parent: null, orbitAngle: 0, orbitRadius: 0, orbitSpeed: 0, vx: 0, vy: 0, connectionCount: 0, totalStrength: 0
        }];
        this.connections = []; this.synapses = []; this.renderGraph(); return;
    }


    // Second pass: Process connections and calculate counts/strengths
    memoryConnections.forEach(conn => {
      const sourceNode = nodeMap.get(conn.source);
      const targetNode = nodeMap.get(conn.target);
      const strength = conn.strength || 0.5; // Default strength if missing

      if (sourceNode && targetNode) {
        const connection = {
          source: sourceNode,
          target: targetNode,
          strength: strength
        };
        this.connections.push(connection);

        // Add connection reference to both nodes involved
        nodeConnections.get(sourceNode).push({ node: targetNode, strength: strength });
        nodeConnections.get(targetNode).push({ node: sourceNode, strength: strength });

        // Increment connection counts and total strength
        sourceNode.connectionCount++;
        targetNode.connectionCount++;
        sourceNode.totalStrength += strength;
        targetNode.totalStrength += strength;
      }
    });

    // Third pass: Determine orbital parents, sizes, and orbit parameters
    const maxConnections = Math.max(1, ...this.nodes.map(n => n.connectionCount)); // Avoid division by zero
    this.nodes.forEach(node => {
      if (node.isCentral) return; // Skip central node

      // --- Determine Size ---
      const sizeRatio = node.connectionCount / maxConnections;
      node.size = this.options.nodeSize.min + (this.options.nodeSize.max - this.options.nodeSize.min) * sizeRatio;
      // Ensure size is at least min
      node.size = Math.max(this.options.nodeSize.min, node.size);

      // --- Determine Orbital Parent ---
      let strongestConnection = null;
      let maxStrength = -1;

      nodeConnections.get(node).forEach(connInfo => {
        if (connInfo.strength > maxStrength) {
          maxStrength = connInfo.strength;
          strongestConnection = connInfo;
        }
      });

      // Assign parent (default to central node if no connections or strongest is central)
      if (strongestConnection && strongestConnection.node !== centralNode) {
          // Check if the potential parent is significantly "larger" (more connected)
          // This prevents small nodes orbiting other small nodes excessively if they have a weak link to center
          const potentialParent = strongestConnection.node;
          if (potentialParent.connectionCount > node.connectionCount + 1 || nodeConnections.get(node).every(c => c.node !== centralNode)) {
             node.parent = potentialParent;
          } else {
             node.parent = centralNode; // Default to central if potential parent isn't significantly larger or if connected to central
          }

      } else {
        node.parent = centralNode; // Default to central node
      }


       // Ensure parent is assigned if node exists but had no connections
       if (!node.parent && centralNode) {
          node.parent = centralNode;
       }


      // --- Determine Orbit Parameters ---
       if (node.parent) {
          const baseRadius = this.options.distanceFactor * (node.parent.isCentral ? 1 : 0.5); // Smaller orbits around non-central parents
          const strengthFactor = maxStrength > 0 ? (1 / (maxStrength + 0.5)) : 1; // Stronger connections orbit closer (adjust divisor)
          node.orbitRadius = baseRadius * strengthFactor * (1 + Math.random() * 0.2); // Add slight variation
          node.orbitRadius = Math.max(node.parent.size + node.size + 10, node.orbitRadius); // Ensure no overlap initially

          const baseSpeed = this.options.orbitSpeed * (node.parent.isCentral ? 1 : 1.5); // Faster orbits around smaller parents? Or keep simple?
          node.orbitSpeed = baseSpeed * (0.8 + Math.random() * 0.4); // Random variation

          // Initial position based on parent and orbit
          node.x = node.parent.x + Math.cos(node.orbitAngle) * node.orbitRadius;
          node.y = node.parent.y + Math.sin(node.orbitAngle) * node.orbitRadius;
      } else {
          // Fallback if somehow no parent assigned (shouldn't happen with centralNode check)
          const angle = Math.random() * 2 * Math.PI;
          const radius = this.options.distanceFactor * 1.5;
          node.x = centerX + Math.cos(angle) * radius;
          node.y = centerY + Math.sin(angle) * radius;
          node.orbitRadius = 0;
          node.orbitSpeed = 0;
       }


    });

    // Create neural activity synapses (can remain similar)
    this.synapses = [];
    if (centralNode) { // Only create synapses if central node exists
        const numSynapses = Math.min(15, Math.ceil(this.nodes.length * 0.8)); // More synapses
        for (let i = 0; i < numSynapses; i++) {
          const startNode = this.nodes[Math.floor(Math.random() * this.nodes.length)];
          // Ensure synapse doesn't start and end at the same node (especially central)
          if (startNode === centralNode && this.nodes.length > 1) {
              i--; // Try again
              continue;
          }
          const synapse = {
            startX: startNode.x,
            startY: startNode.y,
            endX: centralNode.x,
            endY: centralNode.y,
            progress: Math.random(), // Start at random points
            speed: 0.02 + Math.random() * 0.03, // Slightly slower, more variation
            active: Math.random() > 0.8, // Start some active
            cooldown: 10 + Math.floor(Math.random() * 50) // Shorter cooldown
          };
          this.synapses.push(synapse);
        }
    }


    // Render the graph with updated nodes and connections
    this.renderGraph();
    console.log("Mind map initialized from memory data with planetary orbits.");
  }

  // --- NEW: Fetch graph data from API ---
  loadGraphData(centerNodeId = this.options.initialCenterNodeId, depth = this.options.initialDepth) {
      let url = this.options.graphApiUrl;
      const params = new URLSearchParams();
      if (centerNodeId) {
          params.append('center', centerNodeId);
      }
      params.append('depth', depth);
      url += `?${params.toString()}`;

      console.log(`Fetching graph data from: ${url}`);
      fetch(url)
          .then(response => {
              if (!response.ok) {
                  throw new Error(`HTTP error! status: ${response.status}`);
              }
              return response.json();
          })
          .then(data => {
              if (!data || !data.nodes) {
                   console.warn("Received empty or invalid graph data from API.");
                   // Initialize with an empty graph or a default placeholder if needed
                   this.nodes = [];
                   this.connections = [];
                   this.renderGraph(); // Render the empty state
                   return;
              }
              console.log("Graph data received:", data);
              this.initFromGraphData(data);
          })
          .catch(error => {
              console.error("Error loading graph data:", error);
              // Display an error message to the user?
              // Optionally fallback to sample data or an error state visualization
              this.nodes = [{ id: 'error', label: 'Error Loading Data', x: this.width / 2, y: this.height / 2, size: this.options.nodeSize.max, color: 'red', isCentral: true, fixed: true }];
              this.connections = [];
              this.renderGraph();
          });
  }

  // --- NEW: Initialize graph from API data ---
  initFromGraphData(graphData) {
      this.nodes = [];
      this.connections = [];

      if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
          console.warn("No nodes received from the API.");
          // Maybe create a default central node if the DB is truly empty?
          this.nodes.push({
              id: 'initial_node', // Use a placeholder ID
              label: 'Memory Core',
              type: 'Semantic', // Default type
              x: this.width / 2,
              y: this.height / 2,
              vx: 0,
              vy: 0,
              size: this.options.nodeSize.max, // Make it prominent
              isCentral: true, // Mark it as central for styling/behavior
              fixed: true // Keep it centered initially
          });
          this.renderGraph();
          return;
      }


      const nodeMap = new Map();

      // Process nodes
      graphData.nodes.forEach((nodeData, index) => {
          const isCentral = index === 0; // Assume first node is the center or most relevant? Or determine based on ID match?
          const node = {
              id: nodeData.id, // Use the ID from the database
              label: nodeData.label,
              type: nodeData.type, // Added type property
              text: nodeData.text, // Store full text if needed for tooltips etc.
              created_at: nodeData.created_at,
              last_access: nodeData.last_access,
              x: this.width / 2 + (Math.random() - 0.5) * 200, // Initial random placement around center
              y: this.height / 2 + (Math.random() - 0.5) * 200,
              vx: 0,
              vy: 0,
              size: this.options.nodeSize.min + Math.random() * (this.options.nodeSize.max - this.options.nodeSize.min), // Placeholder size logic
              isCentral: isCentral, // Mark the first node as central for now
              fixed: isCentral, // Fix the central node
              children: [],
              parent: null // Will be determined by edges if applicable
          };
          // If it's the central node, place it precisely in the middle
          if (isCentral) {
              node.x = this.width / 2;
              node.y = this.height / 2;
              node.size = this.options.nodeSize.max; // Make central node largest
          }

          this.nodes.push(node);
          nodeMap.set(node.id, node);
      });

       // Calculate node sizes based on connections (degree) after all nodes exist
       this.calculateNodeSizes(graphData.edges, nodeMap);


      // Process edges (connections)
      graphData.edges.forEach(edgeData => {
          const sourceNode = nodeMap.get(edgeData.source);
          const targetNode = nodeMap.get(edgeData.target);

          if (sourceNode && targetNode) {
              this.connections.push({
                  source: sourceNode,
                  target: targetNode,
                  rel_type: edgeData.rel_type, // Store relationship type
                  strength: 1 // Default strength, could vary by rel_type later
              });

              // Basic parent/child assignment (assuming 'updates' or specific 'instanceOf' implies hierarchy)
              // This might need refinement based on edge types and desired structure
              if (edgeData.rel_type === 'updates') { // New node updates old node
                  // Visually, we might not connect 'updates' directly or style them differently
              } else if (!targetNode.parent && !targetNode.fixed) { // Assign first non-fixed connection as parent
                  targetNode.parent = sourceNode;
                  sourceNode.children.push(targetNode);
              } else if (!sourceNode.parent && !sourceNode.fixed) {
                   sourceNode.parent = targetNode;
                   targetNode.children.push(sourceNode);
              }


          } else {
              console.warn(`Edge references non-existent node: ${edgeData.source} -> ${edgeData.target}`);
          }
      });

      // Re-render the graph with the new data
      this.renderGraph();
  }

   // --- NEW: Calculate node sizes based on degree ---
   calculateNodeSizes(edges, nodeMap) {
      const degrees = new Map();
      this.nodes.forEach(node => degrees.set(node.id, 0));

      edges.forEach(edge => {
          degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
          degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
      });

      const maxDegree = Math.max(...degrees.values());

      this.nodes.forEach(node => {
          if (!node.fixed) { // Don't resize the fixed central node based on degree? Or allow it?
              const degree = degrees.get(node.id) || 0;
              const sizeRatio = maxDegree > 0 ? degree / maxDegree : 0;
              node.size = this.options.nodeSize.min + sizeRatio * (this.options.nodeSize.max - this.options.nodeSize.min);
          } else {
               // Ensure central node retains its max size
               node.size = this.options.nodeSize.max;
          }
      });
  }


  // Method to handle node clicks (e.g., recenter the graph)
  handleNodeClick(node) {
    console.log(`Node clicked: ${node.label} (ID: ${node.id}, Type: ${node.type})`);
    // Option 1: Recenter the graph on the clicked node
    // this.loadGraphData(node.id, this.options.initialDepth);

    // Option 2: Display more info about the node (e.g., in a sidebar)
    this.displayNodeDetails(node);

    // Option 3: Add simple interaction - make clicked node pulse briefly?
    if (node.element) {
        const circle = node.element.querySelector('circle');
        if (circle) {
            circle.style.transition = 'transform 0.2s ease-out';
            circle.style.transform = 'scale(1.2)';
            setTimeout(() => {
                circle.style.transform = 'scale(1)';
            }, 200);
        }
    }
  }

  // Placeholder for showing node details (e.g., in a separate panel)
  displayNodeDetails(node) {
      const detailsPanel = document.getElementById('node-details-panel'); // Assuming you have a div with this ID
      if (detailsPanel) {
          detailsPanel.innerHTML = `
              <h3>${node.label}</h3>
              <p><strong>ID:</strong> ${node.id}</p>
              <p><strong>Type:</strong> ${node.type}</p>
              <p><strong>Text:</strong> ${node.text || 'N/A'}</p>
              <p><strong>Created:</strong> ${new Date(node.created_at).toLocaleString()}</p>
              <p><strong>Last Access:</strong> ${new Date(node.last_access).toLocaleString()}</p>
              <!-- Add more details or actions here -->
          `;
          detailsPanel.style.display = 'block'; // Show the panel
      }
  }

  // Placeholder for showing tooltip/info on hover
  showNodeInfo(node, event) {
      let tooltip = document.getElementById('mindmap-tooltip');
      if (!tooltip) {
          tooltip = document.createElement('div');
          tooltip.id = 'mindmap-tooltip';
          tooltip.style.position = 'absolute';
          tooltip.style.background = 'rgba(0,0,0,0.7)';
          tooltip.style.color = 'white';
          tooltip.style.padding = '5px 10px';
          tooltip.style.borderRadius = '4px';
          tooltip.style.fontSize = '12px';
          tooltip.style.pointerEvents = 'none'; // Don't let tooltip interfere with mouse events
          tooltip.style.zIndex = '1000'; // Ensure tooltip is on top
          document.body.appendChild(tooltip);
      }

      tooltip.innerHTML = `<strong>${node.label}</strong> (${node.type})`;
      tooltip.style.left = `${event.clientX + 15}px`; // Position near cursor
      tooltip.style.top = `${event.clientY}px`;
      tooltip.style.display = 'block';
  }

  // Hide tooltip
  hideNodeInfo() {
      const tooltip = document.getElementById('mindmap-tooltip');
      if (tooltip) {
          tooltip.style.display = 'none';
      }
  }
}

// Initialize the mind map when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.mindMap = new MindMap('fred-visualization', {
    centralTopic: "AI Brain",
    dataFile: 'static/memory_map_data.json' // Point to our memory data file
  });
  
  // Handle window resize
  window.addEventListener('resize', () => {
    if (window.mindMap) {
      window.mindMap.resize();
    }
  });
}); 