/**
 * F.R.E.D. Enhanced 3D Solar System Memory Visualization
 * A physics-based 3D solar system with hierarchical orbital relationships
 * and detailed memory interaction
 */

class MemorySolarSystem {
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 2000);
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance"
        });
        
        this.memories = [];
        this.memoryConnections = new Map(); // Store connection data
        this.memoryGroups = new Map();
        this.centralCore = null;
        this.controls = null;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.isActive = false;
        this.animationId = null;
        this.selectedMemory = null;
        this.detailsPanel = null;
        
        // Physics and animation
        this.clock = new THREE.Clock();
        this.time = 0;
        
        // Enhanced orbital system
        this.orbitalHierarchy = new Map(); // nodeid -> { parent: nodeid, children: [nodeids], strength: number }
        this.primaryMemories = []; // Memories that orbit F.R.E.D. directly
        this.secondaryMemories = []; // Memories that orbit primary memories
        
        // Initialize the solar system
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.setupCamera();
        this.setupControls();
        this.setupLighting();
        this.createCentralCore();
        this.setupEventListeners();
        this.createDetailsPanel();
        this.loadMemoryData();
        this.animate();
    }
    
    setupRenderer() {
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Hide loading message
        const loadingElement = document.getElementById('solar-system-loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    setupCamera() {
        this.camera.position.set(0, 50, 200);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupControls() {
        // Create custom orbit controls if Three.OrbitControls is not available
        if (window.THREE && window.THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.maxDistance = 500;
            this.controls.minDistance = 50;
        } else {
            // Simple mouse controls fallback
            this.setupBasicMouseControls();
        }
    }
    
    setupBasicMouseControls() {
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let targetRotationY = 0;
        let targetRotationX = 0;
        let currentRotationY = 0;
        let currentRotationX = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            isMouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (isMouseDown) {
                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;
                
                targetRotationY += deltaX * 0.01;
                targetRotationX += deltaY * 0.01;
                
                mouseX = event.clientX;
                mouseY = event.clientY;
            }
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });
        
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const zoom = event.deltaY * 0.1;
            this.camera.position.multiplyScalar(1 + zoom * 0.01);
            
            // Clamp zoom
            const distance = this.camera.position.length();
            if (distance < 50) {
                this.camera.position.normalize().multiplyScalar(50);
            } else if (distance > 500) {
                this.camera.position.normalize().multiplyScalar(500);
            }
        });
        
        // Update camera rotation in animation loop
        this.updateBasicControls = () => {
            currentRotationY += (targetRotationY - currentRotationY) * 0.05;
            currentRotationX += (targetRotationX - currentRotationX) * 0.05;
            
            // Apply rotation around origin
            const distance = this.camera.position.length();
            this.camera.position.x = distance * Math.sin(currentRotationY) * Math.cos(currentRotationX);
            this.camera.position.y = distance * Math.sin(currentRotationX);
            this.camera.position.z = distance * Math.cos(currentRotationY) * Math.cos(currentRotationX);
            
            this.camera.lookAt(0, 0, 0);
        };
    }
    
    setupLighting() {
        // Ambient light for overall illumination
        const ambientLight = new THREE.AmbientLight(0x404080, 0.4);
        this.scene.add(ambientLight);
        
        // Central light from F.R.E.D.'s core
        const coreLight = new THREE.PointLight(0xCC785C, 2, 400);
        coreLight.position.set(0, 0, 0);
        coreLight.castShadow = true;
        coreLight.shadow.mapSize.width = 1024;
        coreLight.shadow.mapSize.height = 1024;
        this.scene.add(coreLight);
        
        // Secondary accent lights
        const accentLight1 = new THREE.PointLight(0x61AAF2, 1, 300);
        accentLight1.position.set(100, 50, 100);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xD4A27F, 0.8, 250);
        accentLight2.position.set(-80, -30, -80);
        this.scene.add(accentLight2);
    }
    
    createCentralCore() {
        // F.R.E.D.'s central processing core
        const coreGeometry = new THREE.SphereGeometry(15, 32, 32);
        
        // Create a custom shader material for the core
        const coreMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color1: { value: new THREE.Color(0xCC785C) },
                color2: { value: new THREE.Color(0xD4A27F) },
                color3: { value: new THREE.Color(0x61AAF2) }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                void main() {
                    vPosition = position;
                    vNormal = normal;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color1;
                uniform vec3 color2;
                uniform vec3 color3;
                varying vec3 vPosition;
                varying vec3 vNormal;
                
                void main() {
                    float pulse = sin(time * 2.0) * 0.5 + 0.5;
                    float noise = sin(vPosition.x * 10.0 + time) * sin(vPosition.y * 10.0 + time) * 0.1;
                    
                    vec3 color = mix(color1, color2, pulse + noise);
                    color = mix(color, color3, sin(time * 0.5) * 0.3 + 0.3);
                    
                    float fresnel = 1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0));
                    color += fresnel * 0.2;
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `
        });
        
        this.centralCore = new THREE.Mesh(coreGeometry, coreMaterial);
        this.centralCore.userData = {
            type: 'core',
            label: 'F.R.E.D. Central Processing Core',
            description: 'The heart of F.R.E.D.\'s consciousness and memory system',
            connections: [],
            text: 'Central processing unit managing all memories and cognitive functions'
        };
        this.scene.add(this.centralCore);
        
        // Add a subtle glow effect
        const glowGeometry = new THREE.SphereGeometry(18, 16, 16);
        const glowMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                glowColor: { value: new THREE.Color(0xCC785C) }
            },
            vertexShader: `
                varying vec3 vNormal;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 glowColor;
                varying vec3 vNormal;
                
                void main() {
                    float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
                    intensity *= (sin(time * 3.0) * 0.3 + 0.7);
                    gl_FragColor = vec4(glowColor, intensity * 0.3);
                }
            `,
            side: THREE.BackSide,
            blending: THREE.AdditiveBlending,
            transparent: true
        });
        
        const coreGlow = new THREE.Mesh(glowGeometry, glowMaterial);
        this.scene.add(coreGlow);
    }
    
    createDetailsPanel() {
        // Create a floating details panel for memory information
        this.detailsPanel = document.createElement('div');
        this.detailsPanel.className = 'memory-details-panel';
        this.detailsPanel.innerHTML = `
            <div class="memory-details-header">
                <h3 class="memory-title">Memory Details</h3>
                <button class="memory-close-btn" onclick="window.mindMap?.hideMemoryDetails()">&times;</button>
            </div>
            <div class="memory-details-content">
                <div class="memory-basic-info">
                    <div class="memory-label"></div>
                    <div class="memory-type"></div>
                    <div class="memory-importance"></div>
                </div>
                <div class="memory-text-content">
                    <h4>Content</h4>
                    <p class="memory-text"></p>
                </div>
                <div class="memory-connections-section">
                    <h4>Connections</h4>
                    <div class="memory-connections-list"></div>
                </div>
                <div class="memory-metadata">
                    <div class="memory-created"></div>
                    <div class="memory-accessed"></div>
                </div>
            </div>
        `;
        this.detailsPanel.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            width: 320px;
            max-width: calc(100vw - 40px);
            max-height: calc(100vh - 120px);
            background: rgba(25, 25, 25, 0.98);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(212, 162, 127, 0.3);
            border-radius: 12px;
            color: #FAFAF7;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7);
            z-index: 9999;
            display: none;
            overflow-y: auto;
        `;
        
        this.container.parentElement.appendChild(this.detailsPanel);
        
        // Add CSS for panel styling
        if (!document.getElementById('memory-details-styles')) {
            const styles = document.createElement('style');
            styles.id = 'memory-details-styles';
            styles.textContent = `
                .memory-details-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 16px 20px;
                    border-bottom: 1px solid rgba(212, 162, 127, 0.2);
                    background: rgba(204, 120, 92, 0.1);
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                .memory-title {
                    margin: 0;
                    font-size: 16px;
                    font-weight: 600;
                    color: #CC785C;
                    flex: 1;
                    margin-right: 12px;
                }
                .memory-close-btn {
                    background: none;
                    border: none;
                    color: #BF4D43;
                    font-size: 24px;
                    cursor: pointer;
                    padding: 8px;
                    border-radius: 6px;
                    transition: all 0.2s ease;
                    line-height: 1;
                    min-width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-shrink: 0;
                }
                .memory-close-btn:hover {
                    background: rgba(191, 77, 67, 0.2);
                    color: #E53E3E;
                    transform: scale(1.1);
                }
                .memory-details-content {
                    padding: 20px;
                }
                .memory-basic-info {
                    margin-bottom: 20px;
                }
                .memory-label {
                    font-size: 18px;
                    font-weight: 600;
                    color: #D4A27F;
                    margin-bottom: 8px;
                }
                .memory-type {
                    font-size: 12px;
                    text-transform: uppercase;
                    color: #61AAF2;
                    margin-bottom: 4px;
                }
                .memory-importance {
                    font-size: 11px;
                    color: #9DA7B3;
                }
                .memory-text-content {
                    margin-bottom: 20px;
                }
                .memory-text-content h4 {
                    margin: 0 0 8px 0;
                    font-size: 14px;
                    color: #CC785C;
                }
                .memory-text {
                    margin: 0;
                    line-height: 1.5;
                    color: #CED3D9;
                }
                .memory-connections-section {
                    margin-bottom: 20px;
                }
                .memory-connections-section h4 {
                    margin: 0 0 12px 0;
                    font-size: 14px;
                    color: #CC785C;
                }
                .memory-connections-list {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .memory-connection {
                    padding: 8px 12px;
                    background: rgba(97, 170, 242, 0.1);
                    border: 1px solid rgba(97, 170, 242, 0.2);
                    border-radius: 6px;
                    font-size: 12px;
                }
                .connection-label {
                    font-weight: 600;
                    color: #61AAF2;
                }
                .connection-type {
                    color: #9DA7B3;
                    font-style: italic;
                }
                .memory-metadata {
                    border-top: 1px solid rgba(212, 162, 127, 0.2);
                    padding-top: 16px;
                    font-size: 11px;
                    color: #9DA7B3;
                }
                .memory-metadata div {
                    margin-bottom: 4px;
                }
            `;
            document.head.appendChild(styles);
        }
    }
    
    async loadMemoryData() {
        try {
            const response = await fetch('/api/memory_visualization_data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const memoryData = await response.json();
            
            // Also fetch detailed connection data for each memory
            await this.loadConnectionData(memoryData);
            
            this.createHierarchicalMemorySystem(memoryData);
        } catch (error) {
            console.warn('Failed to load memory data:', error);
            this.createDemoMemories();
        }
    }
    
    async loadConnectionData(memoryData) {
        // Load detailed connection information for each memory
        console.log('Loading connection data for', memoryData.length, 'memories');
        for (const memory of memoryData) {
            try {
                const response = await fetch(`/api/memory/${memory.nodeid}/connections`);
                if (response.ok) {
                    const connectionData = await response.json();
                    console.log(`Loaded connections for memory ${memory.nodeid}:`, connectionData);
                    this.memoryConnections.set(memory.nodeid.toString(), connectionData);
                } else {
                    console.warn(`Failed to load connections for memory ${memory.nodeid}: HTTP ${response.status}`);
                }
            } catch (error) {
                console.warn(`Failed to load connections for memory ${memory.nodeid}:`, error);
                this.memoryConnections.set(memory.nodeid.toString(), { connections: [] });
            }
        }
        console.log('Finished loading connection data. Total connections loaded:', this.memoryConnections.size);
    }
    
    createHierarchicalMemorySystem(memoryData) {
        if (!memoryData || memoryData.length === 0) {
            this.createDemoMemories();
            return;
        }
        
        // Sort memories by total connections (importance)
        const sortedMemories = [...memoryData].sort((a, b) => b.total_edge_count - a.total_edge_count);
        
        // Determine hierarchical relationships
        this.buildOrbitalHierarchy(sortedMemories);
        
        // Create primary memories (orbit F.R.E.D. directly)
        this.createPrimaryMemories(this.primaryMemories);
        
        // Create secondary memories (orbit primary memories)
        this.createSecondaryMemories(this.secondaryMemories);
        
        // Create connection lines between related memories
        this.createConnectionLines();
    }
    
    buildOrbitalHierarchy(sortedMemories) {
        // Determine which memories should orbit F.R.E.D. directly vs orbit other memories
        const totalMemories = sortedMemories.length;
        const primaryCount = Math.min(8, Math.ceil(totalMemories * 0.3)); // Top 30% or max 8
        
        // Primary memories - highest connection counts
        this.primaryMemories = sortedMemories.slice(0, primaryCount);
        
        // Secondary memories - remaining memories
        const remainingMemories = sortedMemories.slice(primaryCount);
        
        // Assign secondary memories to orbit their strongest connected primary memory
        this.secondaryMemories = [];
        
        for (const memory of remainingMemories) {
            let bestParent = null;
            let strongestConnection = 0;
            
            // Find the primary memory this memory has the strongest connection to
            const connections = this.memoryConnections.get(memory.nodeid.toString())?.connections || [];
            
            for (const connection of connections) {
                const connectedNodeId = connection.direction === 'outgoing' ? 
                    connection.target_nodeid : connection.source_nodeid;
                
                // Check if this connected node is a primary memory
                const primaryMemory = this.primaryMemories.find(p => p.nodeid.toString() === connectedNodeId.toString());
                if (primaryMemory) {
                    // Use relationship type to determine connection strength
                    const relationshipStrength = this.getRelationshipStrength(connection.rel_type);
                    if (relationshipStrength > strongestConnection) {
                        strongestConnection = relationshipStrength;
                        bestParent = primaryMemory;
                    }
                }
            }
            
            // If no strong connection to primary memories, assign to most connected primary
            if (!bestParent && this.primaryMemories.length > 0) {
                bestParent = this.primaryMemories[0]; // Most connected primary memory
            }
            
            if (bestParent) {
                memory.orbitalParent = bestParent;
                this.secondaryMemories.push(memory);
                
                // Track this relationship in hierarchy
                this.orbitalHierarchy.set(memory.nodeid.toString(), {
                    parent: bestParent.nodeid.toString(),
                    children: [],
                    strength: strongestConnection
                });
                
                // Add to parent's children list
                const parentHierarchy = this.orbitalHierarchy.get(bestParent.nodeid.toString()) || { children: [] };
                parentHierarchy.children.push(memory.nodeid.toString());
                this.orbitalHierarchy.set(bestParent.nodeid.toString(), parentHierarchy);
            }
        }
    }
    
    getRelationshipStrength(relType) {
        // Assign strength values to different relationship types
        const strengthMap = {
            'contains': 10,
            'partOf': 9,
            'dependsOn': 8,
            'enablesGoal': 7,
            'relatedTo': 6,
            'instanceOf': 5,
            'updates': 4,
            'precedes': 3,
            'causes': 2,
            'locatedAt': 1
        };
        return strengthMap[relType] || 1;
    }
    
    createPrimaryMemories(primaryMemories) {
        // Create memories that orbit F.R.E.D. directly
        const baseRadius = 60;
        const radiusIncrement = 25;
        
        primaryMemories.forEach((memory, index) => {
            const ringIndex = Math.floor(index / 4); // 4 memories per ring
            const angleIndex = index % 4;
            const radius = baseRadius + (ringIndex * radiusIncrement);
            const angle = (angleIndex / 4) * Math.PI * 2;
            
            const memoryOrb = this.createMemoryOrb(memory, {
                radius: radius,
                speed: 0.015 - (ringIndex * 0.003), // Outer rings slower
                angle: angle,
                isPrimary: true
            });
            
            memoryOrb.userData.orbitalRadius = radius;
            memoryOrb.userData.orbitalSpeed = 0.015 - (ringIndex * 0.003);
            memoryOrb.userData.orbitalAngle = angle;
            memoryOrb.userData.isPrimary = true;
            
            this.memories.push(memoryOrb);
        });
    }
    
    createSecondaryMemories(secondaryMemories) {
        // Create memories that orbit primary memories
        const satelliteRadius = 20; // Distance from parent memory
        
        // Group secondary memories by their parent
        const parentGroups = new Map();
        secondaryMemories.forEach(memory => {
            const parentId = memory.orbitalParent.nodeid.toString();
            if (!parentGroups.has(parentId)) {
                parentGroups.set(parentId, []);
            }
            parentGroups.get(parentId).push(memory);
        });
        
        // Create secondary memories around their parents
        parentGroups.forEach((children, parentId) => {
            children.forEach((memory, index) => {
                const angleStep = (Math.PI * 2) / Math.max(children.length, 3);
                const angle = index * angleStep;
                
                const memoryOrb = this.createMemoryOrb(memory, {
                    radius: satelliteRadius,
                    speed: 0.02 + (index * 0.005), // Varied speeds for visual interest
                    angle: angle,
                    isPrimary: false,
                    parentId: parentId
                });
                
                memoryOrb.userData.orbitalRadius = satelliteRadius;
                memoryOrb.userData.orbitalSpeed = 0.02 + (index * 0.005);
                memoryOrb.userData.orbitalAngle = angle;
                memoryOrb.userData.isPrimary = false;
                memoryOrb.userData.parentId = parentId;
                
                this.memories.push(memoryOrb);
            });
        });
    }
    
    createMemoryOrb(memoryData, orbitalConfig) {
        // Memory orb size based on importance
        const importance = memoryData.total_edge_count || 1;
        
        // Safely calculate maxImportance to avoid NaN
        const existingImportances = this.memories.map(m => m.userData.importance || 1);
        const maxImportance = Math.max(...existingImportances, importance, 15);
        const safeMaxImportance = isNaN(maxImportance) || maxImportance <= 0 ? 15 : maxImportance;
        
        const baseSize = orbitalConfig.isPrimary ? 3 : 2;
        const orbSize = baseSize + (importance / safeMaxImportance) * (orbitalConfig.isPrimary ? 8 : 4);
        
        // Memory type determines color and material
        const memoryType = memoryData.type || 'general';
        const colorScheme = this.getMemoryColorScheme(memoryType);
        
        // Create orb geometry
        const orbGeometry = new THREE.SphereGeometry(orbSize, 16, 16);
        
        // Create memory-specific material with enhanced effects
        const orbMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                primaryColor: { value: colorScheme.primary },
                secondaryColor: { value: colorScheme.secondary },
                importance: { value: importance / maxImportance },
                pulseSpeed: { value: colorScheme.pulseSpeed },
                isSelected: { value: 0.0 }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                void main() {
                    vPosition = position;
                    vNormal = normal;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 primaryColor;
                uniform vec3 secondaryColor;
                uniform float importance;
                uniform float pulseSpeed;
                uniform float isSelected;
                varying vec3 vPosition;
                varying vec3 vNormal;
                
                void main() {
                    float pulse = sin(time * pulseSpeed + importance * 10.0) * 0.5 + 0.5;
                    vec3 color = mix(primaryColor, secondaryColor, pulse);
                    
                    // Add surface detail
                    float noise = sin(vPosition.x * 5.0) * sin(vPosition.y * 5.0) * sin(vPosition.z * 5.0);
                    color += noise * 0.1;
                    
                    // Fresnel effect for depth
                    float fresnel = 1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0));
                    color += fresnel * 0.3 * importance;
                    
                    // Selection highlight
                    if (isSelected > 0.5) {
                        color += vec3(0.3, 0.3, 0.3) * sin(time * 5.0) * 0.5;
                    }
                    
                    gl_FragColor = vec4(color, 0.9);
                }
            `,
            transparent: true
        });
        
        const memoryOrb = new THREE.Mesh(orbGeometry, orbMaterial);
        
        // Set initial position with validation
        if (orbitalConfig.isPrimary) {
            // Position around F.R.E.D. core
            const radius = orbitalConfig.radius || 60;
            const angle = orbitalConfig.angle || 0;
            
            const x = radius * Math.cos(angle);
            const z = radius * Math.sin(angle);
            const y = (Math.random() - 0.5) * 10; // Slight vertical variation
            
            // Validate calculated positions
            memoryOrb.position.x = isNaN(x) ? 0 : x;
            memoryOrb.position.z = isNaN(z) ? 0 : z;
            memoryOrb.position.y = isNaN(y) ? 0 : y;
        } else {
            // Position will be set relative to parent in animation loop
            memoryOrb.position.set(0, 0, 0);
        }
        
        // Store comprehensive memory data
        memoryOrb.userData = {
            ...memoryData,
            type: 'memory',
            importance: importance,
            colorScheme: colorScheme,
            originalY: memoryOrb.position.y,
            connections: this.memoryConnections.get(memoryData.nodeid.toString())?.connections || [],
            ...orbitalConfig
        };
        
        this.scene.add(memoryOrb);
        
        // Add orbital trail for primary memories
        if (orbitalConfig.isPrimary) {
            this.createOrbitalTrail(memoryOrb);
        }
        
        return memoryOrb;
    }
    
    getMemoryColorScheme(type) {
        const schemes = {
            'conversation': {
                primary: new THREE.Color(0x61AAF2),
                secondary: new THREE.Color(0x9DA7B3),
                pulseSpeed: 2.0
            },
            'knowledge': {
                primary: new THREE.Color(0xD4A27F),
                secondary: new THREE.Color(0xEBDDBC),
                pulseSpeed: 1.5
            },
            'task': {
                primary: new THREE.Color(0xCC785C),
                secondary: new THREE.Color(0xBF4D43),
                pulseSpeed: 3.0
            },
            'emotion': {
                primary: new THREE.Color(0xF0F0EB),
                secondary: new THREE.Color(0xE5E4DF),
                pulseSpeed: 2.5
            },
            'general': {
                primary: new THREE.Color(0xCED3D9),
                secondary: new THREE.Color(0xE4E8ED),
                pulseSpeed: 1.0
            }
        };
        
        return schemes[type] || schemes.general;
    }
    
    createOrbitalTrail(memoryOrb) {
        const points = [];
        const radius = memoryOrb.userData.orbitalRadius || 60;
        const originalY = memoryOrb.userData.originalY || 0;
        
        // Validate radius to prevent NaN
        if (isNaN(radius) || radius <= 0) {
            console.warn('Invalid orbital radius for trail, skipping trail creation');
            return;
        }
        
        for (let i = 0; i <= 64; i++) {
            const angle = (i / 64) * Math.PI * 2;
            const x = radius * Math.cos(angle);
            const z = radius * Math.sin(angle);
            
            // Validate calculated positions
            if (!isNaN(x) && !isNaN(originalY) && !isNaN(z)) {
                points.push(new THREE.Vector3(x, originalY, z));
            }
        }
        
        if (points.length === 0) {
            console.warn('No valid points for orbital trail, skipping creation');
            return;
        }
        
        const trailGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const trailMaterial = new THREE.LineBasicMaterial({
            color: memoryOrb.userData.colorScheme.primary,
            opacity: 0.1,
            transparent: true
        });
        
        const trail = new THREE.Line(trailGeometry, trailMaterial);
        this.scene.add(trail);
    }
    
    createConnectionLines() {
        // Create visible connection lines between strongly related memories
        const connectionLines = [];
        
        this.memories.forEach(memory => {
            const connections = memory.userData.connections || [];
            
            connections.forEach(connection => {
                // Only show strong connections to avoid visual clutter
                const strength = this.getRelationshipStrength(connection.rel_type);
                if (strength >= 6) { // relatedTo and above
                    const connectedNodeId = connection.direction === 'outgoing' ? 
                        connection.target_nodeid : connection.source_nodeid;
                    
                    const connectedMemory = this.memories.find(m => 
                        m.userData.nodeid.toString() === connectedNodeId.toString()
                    );
                    
                    if (connectedMemory) {
                        this.createConnectionLine(memory, connectedMemory, connection.rel_type, strength);
                    }
                }
            });
        });
    }
    
    createConnectionLine(memory1, memory2, relType, strength) {
        // Create a line between two memories
        const points = [
            memory1.position.clone(),
            memory2.position.clone()
        ];
        
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
        
        // Color based on relationship type
        const color = this.getRelationshipColor(relType);
        const opacity = (strength / 10) * 0.3; // Stronger connections more visible
        
        const lineMaterial = new THREE.LineBasicMaterial({
            color: color,
            opacity: opacity,
            transparent: true
        });
        
        const connectionLine = new THREE.Line(lineGeometry, lineMaterial);
        connectionLine.userData = {
            type: 'connection',
            memory1: memory1,
            memory2: memory2,
            relType: relType,
            strength: strength
        };
        
        this.scene.add(connectionLine);
    }
    
    getRelationshipColor(relType) {
        // Assign colors to different relationship types
        const colorMap = {
            'contains': 0xCC785C,      // Fred book cloth
            'partOf': 0xD4A27F,        // Fred kraft  
            'dependsOn': 0x61AAF2,     // Fred focus
            'enablesGoal': 0xBF4D43,   // Fred error
            'relatedTo': 0x9DA7B3,     // Fred cloud dark
            'instanceOf': 0xCED3D9,    // Fred cloud medium
            'updates': 0xE4E8ED,       // Fred cloud light
            'precedes': 0x40403E,      // Fred slate light
            'causes': 0x262625,        // Fred slate medium
            'locatedAt': 0x191919      // Fred slate dark
        };
        return colorMap[relType] || 0x9DA7B3;
    }
    
    createDemoMemories() {
        const demoData = [
            { nodeid: 'demo1', label: 'User Preferences', type: 'knowledge', total_edge_count: 15 },
            { nodeid: 'demo2', label: 'Recent Conversation', type: 'conversation', total_edge_count: 12 },
            { nodeid: 'demo3', label: 'Task Management', type: 'task', total_edge_count: 8 },
            { nodeid: 'demo4', label: 'Learning Patterns', type: 'knowledge', total_edge_count: 10 },
            { nodeid: 'demo5', label: 'Emotional Context', type: 'emotion', total_edge_count: 6 },
            { nodeid: 'demo6', label: 'System State', type: 'general', total_edge_count: 4 }
        ];
        
        this.createMemoryOrbs(demoData);
    }
    
    setupEventListeners() {
        // Mouse interaction
        this.renderer.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.renderer.domElement.addEventListener('click', this.onMouseClick.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.onKeyDown.bind(this));
        
        // Window resize
        window.addEventListener('resize', this.onWindowResize.bind(this));
    }
    
    onKeyDown(event) {
        // ESC key closes the details panel
        if (event.key === 'Escape' && this.detailsPanel && this.detailsPanel.style.display !== 'none') {
            this.hideMemoryDetails();
            event.preventDefault();
        }
    }
    
    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Raycast for hover effects
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.memories);
        
        // Reset all memory hover states
        this.memories.forEach(memory => {
            if (memory.userData.isHovered) {
                memory.scale.setScalar(1);
                memory.userData.isHovered = false;
            }
        });
        
        // Apply hover effect to intersected memory
        if (intersects.length > 0) {
            const memory = intersects[0].object;
            memory.scale.setScalar(1.2);
            memory.userData.isHovered = true;
            this.renderer.domElement.style.cursor = 'pointer';
        } else {
            this.renderer.domElement.style.cursor = 'default';
        }
    }
    
    onMouseClick(event) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects([this.centralCore, ...this.memories]);
        
        if (intersects.length > 0) {
            const clicked = intersects[0].object;
            this.showMemoryDetails(clicked.userData);
        }
    }
    
    showMemoryDetails(memoryData) {
        if (!this.detailsPanel) return;
        
        this.selectedMemory = memoryData;
        
        // Debug: log the memory data structure to understand what we're working with
        console.log('Memory data for details panel:', memoryData);
        
        // Update panel content
        const panel = this.detailsPanel;
        panel.querySelector('.memory-label').textContent = memoryData.label || 'Unknown Memory';
        panel.querySelector('.memory-type').textContent = `Type: ${memoryData.type || 'Unknown'}`;
        panel.querySelector('.memory-importance').textContent = 
            `Connections: ${memoryData.total_edge_count || 0} | Importance: ${Math.round((memoryData.importance || 0) * 100)}%`;
        
        // Update text content - check multiple possible fields
        const textElement = panel.querySelector('.memory-text');
        const contentText = memoryData.text || memoryData.content || memoryData.description || memoryData.label || 'No detailed content available.';
        textElement.textContent = contentText;
        
        // Update connections list - get connections from the stored connection data
        const connectionsList = panel.querySelector('.memory-connections-list');
        connectionsList.innerHTML = '';
        
        // Try to get connections from multiple sources
        let connections = [];
        
        // First try the directly stored connections
        if (memoryData.connections && Array.isArray(memoryData.connections)) {
            connections = memoryData.connections;
        }
        // Then try the connection map
        else if (this.memoryConnections.has(memoryData.nodeid.toString())) {
            const connectionData = this.memoryConnections.get(memoryData.nodeid.toString());
            connections = connectionData.connections || [];
        }
        
        console.log(`Found ${connections.length} connections for memory ${memoryData.nodeid}:`, connections);
        
        if (connections.length > 0) {
            connections.slice(0, 10).forEach(connection => { // Show max 10 connections
                const connectionDiv = document.createElement('div');
                connectionDiv.className = 'memory-connection';
                
                // Handle different connection data formats
                let label, type, direction;
                
                if (connection.target_label || connection.source_label) {
                    // API format
                    label = connection.direction === 'outgoing' ? 
                        (connection.target_label || `Node ${connection.target_nodeid}`) : 
                        (connection.source_label || `Node ${connection.source_nodeid}`);
                    type = connection.rel_type || 'relatedTo';
                    direction = connection.direction || 'unknown';
                } else if (connection.label) {
                    // Direct format
                    label = connection.label;
                    type = connection.type || connection.rel_type || 'relatedTo';
                    direction = connection.direction || 'bidirectional';
                } else {
                    // Fallback
                    label = connection.nodeid || connection.id || 'Unknown Connection';
                    type = connection.rel_type || connection.type || 'relatedTo';
                    direction = connection.direction || 'unknown';
                }
                
                connectionDiv.innerHTML = `
                    <div class="connection-label">${label}</div>
                    <div class="connection-type">${type} (${direction})</div>
                `;
                connectionsList.appendChild(connectionDiv);
            });
            
            if (connections.length > 10) {
                const moreDiv = document.createElement('div');
                moreDiv.className = 'memory-connection';
                moreDiv.innerHTML = `<div class="connection-label">... and ${connections.length - 10} more connections</div>`;
                connectionsList.appendChild(moreDiv);
            }
        } else {
            connectionsList.innerHTML = '<div class="memory-connection">No connections found</div>';
        }
        
        // Update metadata - handle different date formats
        const formatDate = (dateValue) => {
            if (!dateValue) return 'Unknown';
            try {
                if (typeof dateValue === 'string') {
                    return new Date(dateValue).toLocaleDateString();
                } else if (dateValue instanceof Date) {
                    return dateValue.toLocaleDateString();
                } else {
                    return 'Unknown';
                }
            } catch (e) {
                return 'Unknown';
            }
        };
        
        panel.querySelector('.memory-created').textContent = 
            `Created: ${formatDate(memoryData.created_at || memoryData.created)}`;
        panel.querySelector('.memory-accessed').textContent = 
            `Last Accessed: ${formatDate(memoryData.last_access || memoryData.accessed || memoryData.last_accessed)}`;
        
        // Show panel
        panel.style.display = 'block';
        
        // Highlight the selected memory in the visualization
        this.highlightMemory(memoryData.nodeid);
    }
    
    hideMemoryDetails() {
        if (this.detailsPanel) {
            this.detailsPanel.style.display = 'none';
        }
        this.selectedMemory = null;
        this.clearHighlights();
    }
    
    highlightMemory(nodeId) {
        // Clear previous highlights
        this.clearHighlights();
        
        // Find and highlight the selected memory
        const memory = this.memories.find(m => m.userData.nodeid.toString() === nodeId.toString());
        if (memory && memory.material.uniforms) {
            memory.material.uniforms.isSelected.value = 1.0;
            memory.scale.setScalar(1.3);
        }
    }
    
    clearHighlights() {
        this.memories.forEach(memory => {
            if (memory.material.uniforms) {
                memory.material.uniforms.isSelected.value = 0.0;
            }
            memory.scale.setScalar(1.0);
        });
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    animate() {
        this.animationId = requestAnimationFrame(this.animate.bind(this));
        
        const delta = this.clock.getDelta();
        this.time += delta;
        
        // Update central core animation
        if (this.centralCore) {
            this.centralCore.material.uniforms.time.value = this.time;
            this.centralCore.rotation.y += delta * 0.2;
            
            // Find core glow (second child)
            const coreGlow = this.scene.children.find(child => 
                child.material && child.material.uniforms && child.material.uniforms.time
            );
            if (coreGlow && coreGlow !== this.centralCore) {
                coreGlow.material.uniforms.time.value = this.time;
            }
        }
        
        // Update memory orb positions and animations with hierarchical system
        this.memories.forEach(memory => {
            const userData = memory.userData;
            
            if (userData.isPrimary) {
                // Primary memories orbit F.R.E.D. core
                userData.orbitalAngle = (userData.orbitalAngle || 0) + (userData.orbitalSpeed || 0.015) * delta;
                
                const radius = userData.orbitalRadius || 60;
                const angle = userData.orbitalAngle || 0;
                const originalY = userData.originalY || 0;
                
                memory.position.x = radius * Math.cos(angle);
                memory.position.z = radius * Math.sin(angle);
                memory.position.y = originalY + Math.sin(this.time + angle) * 2;
            } else {
                // Secondary memories orbit their parent primary memory
                const parentId = userData.parentId;
                const parentMemory = this.memories.find(m => 
                    m.userData.nodeid && m.userData.nodeid.toString() === parentId && m.userData.isPrimary
                );
                
                if (parentMemory && parentMemory.position) {
                    // Update orbital angle around parent
                    userData.orbitalAngle = (userData.orbitalAngle || 0) + (userData.orbitalSpeed || 0.02) * delta;
                    
                    const radius = userData.orbitalRadius || 20;
                    const angle = userData.orbitalAngle || 0;
                    
                    // Calculate position relative to parent
                    const relativeX = radius * Math.cos(angle);
                    const relativeZ = radius * Math.sin(angle);
                    const relativeY = Math.sin(this.time * 2 + angle) * 3;
                    
                    // Set absolute position based on parent position (with safety checks)
                    memory.position.x = (parentMemory.position.x || 0) + relativeX;
                    memory.position.y = (parentMemory.position.y || 0) + relativeY;
                    memory.position.z = (parentMemory.position.z || 0) + relativeZ;
                } else {
                    // Fallback positioning if parent not found
                    memory.position.x = memory.position.x || 0;
                    memory.position.y = memory.position.y || 0;
                    memory.position.z = memory.position.z || 0;
                }
            }
            
            // Update material animation for all memories
            if (memory.material.uniforms) {
                memory.material.uniforms.time.value = this.time;
            }
            
            // Add subtle rotation for visual interest
            memory.rotation.y += delta * 0.5;
            memory.rotation.x += delta * 0.2;
        });
        
        // Update connection lines to follow memory positions
        this.scene.children.forEach(child => {
            if (child.userData.type === 'connection') {
                const memory1 = child.userData.memory1;
                const memory2 = child.userData.memory2;
                
                if (memory1 && memory2 && memory1.position && memory2.position) {
                    // Update line geometry to follow memory positions with validation
                    const positions = child.geometry.attributes.position.array;
                    const pos1x = memory1.position.x || 0;
                    const pos1y = memory1.position.y || 0;
                    const pos1z = memory1.position.z || 0;
                    const pos2x = memory2.position.x || 0;
                    const pos2y = memory2.position.y || 0;
                    const pos2z = memory2.position.z || 0;
                    
                    // Only update if positions are valid numbers
                    if (!isNaN(pos1x) && !isNaN(pos1y) && !isNaN(pos1z) && 
                        !isNaN(pos2x) && !isNaN(pos2y) && !isNaN(pos2z)) {
                        positions[0] = pos1x;
                        positions[1] = pos1y;
                        positions[2] = pos1z;
                        positions[3] = pos2x;
                        positions[4] = pos2y;
                        positions[5] = pos2z;
                        child.geometry.attributes.position.needsUpdate = true;
                    }
                }
            }
        });
        
        // Update controls
        if (this.controls) {
            this.controls.update();
        } else if (this.updateBasicControls) {
            this.updateBasicControls();
        }
        
        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }
    
    setActive(active) {
        this.isActive = active;
        if (active) {
            // Increase activity when F.R.E.D. is processing
            this.memories.forEach(memory => {
                if (memory.material.uniforms) {
                    memory.material.uniforms.pulseSpeed.value *= 2;
                }
            });
        } else {
            // Reset to normal activity
            this.memories.forEach(memory => {
                if (memory.material.uniforms) {
                    memory.material.uniforms.pulseSpeed.value = memory.userData.colorScheme.pulseSpeed;
                }
            });
        }
    }
    
    resize() {
        this.onWindowResize();
    }
    
    reset() {
        // Reset camera position
        this.camera.position.set(0, 50, 200);
        this.camera.lookAt(0, 0, 0);
        if (this.controls) {
            this.controls.reset();
        }
    }
    
    center() {
        // Smooth camera movement to center
        if (this.controls) {
            this.controls.reset();
        }
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Clean up Three.js resources
        this.scene.traverse((object) => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const container = document.querySelector('.visualization-container');
    if (container) {
        // Load Three.js from CDN if not already loaded
        if (typeof THREE === 'undefined') {
            const loadThreeJS = (urls, index = 0) => {
                if (index >= urls.length) {
                    console.error('All Three.js CDN sources failed to load');
                    createFallbackVisualization(container);
                    return;
                }
                
                const script = document.createElement('script');
                script.src = urls[index];
                
                script.onload = () => {
                    console.log(`Three.js loaded successfully from ${urls[index]}`);
                    window.mindMap = new MemorySolarSystem(container);
                };
                
                script.onerror = () => {
                    console.warn(`Failed to load Three.js from ${urls[index]}, trying next source...`);
                    loadThreeJS(urls, index + 1);
                };
                
                document.head.appendChild(script);
            };
            
            // Multiple CDN sources for reliability
            const cdnUrls = [
                'https://cdn.jsdelivr.net/npm/three@0.152.0/build/three.min.js',
                'https://unpkg.com/three@0.152.0/build/three.min.js',
                'https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js'
            ];
            
            loadThreeJS(cdnUrls);
        } else {
            console.log('Three.js already loaded, initializing solar system');
            window.mindMap = new MemorySolarSystem(container);
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.mindMap) {
        window.mindMap.destroy();
    }
});

// Fallback 2D visualization if Three.js fails to load
function createFallbackVisualization(container) {
    console.log('Creating fallback 2D memory visualization');
    
    // Create canvas for 2D visualization
    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.borderRadius = 'var(--radius-md)';
    
    const ctx = canvas.getContext('2d');
    
    // Memory data for demo
    const memories = [
        { x: 200, y: 150, size: 20, color: '#61AAF2', label: 'Conversations', angle: 0 },
        { x: 300, y: 200, size: 15, color: '#D4A27F', label: 'Knowledge', angle: Math.PI / 3 },
        { x: 250, y: 250, size: 18, color: '#CC785C', label: 'Tasks', angle: Math.PI },
        { x: 180, y: 220, size: 12, color: '#F0F0EB', label: 'Emotions', angle: Math.PI * 1.5 },
        { x: 320, y: 170, size: 14, color: '#CED3D9', label: 'General', angle: Math.PI / 6 }
    ];
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    let animationTime = 0;
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw background gradient
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 200);
        gradient.addColorStop(0, 'rgba(97, 170, 242, 0.1)');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw central core
        const coreRadius = 25 + Math.sin(animationTime * 0.02) * 3;
        const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreRadius);
        coreGradient.addColorStop(0, '#CC785C');
        coreGradient.addColorStop(0.7, '#D4A27F');
        coreGradient.addColorStop(1, 'rgba(204, 120, 92, 0.3)');
        
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw core label
        ctx.fillStyle = '#F0F0EB';
        ctx.font = '12px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('F.R.E.D. CORE', centerX, centerY + 40);
        
        // Draw memory orbs
        memories.forEach((memory, index) => {
            const orbitRadius = 80 + index * 20;
            const orbitSpeed = 0.001 + index * 0.0005;
            
            // Calculate orbital position
            const angle = memory.angle + animationTime * orbitSpeed;
            const x = centerX + Math.cos(angle) * orbitRadius;
            const y = centerY + Math.sin(angle) * orbitRadius;
            
            // Draw orbital trail
            ctx.strokeStyle = memory.color + '20';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(centerX, centerY, orbitRadius, 0, Math.PI * 2);
            ctx.stroke();
            
            // Draw memory orb
            const pulse = Math.sin(animationTime * 0.03 + index) * 0.3 + 0.7;
            const orbSize = memory.size * pulse;
            
            const orbGradient = ctx.createRadialGradient(x, y, 0, x, y, orbSize);
            orbGradient.addColorStop(0, memory.color);
            orbGradient.addColorStop(1, memory.color + '60');
            
            ctx.fillStyle = orbGradient;
            ctx.beginPath();
            ctx.arc(x, y, orbSize, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw memory label
            ctx.fillStyle = '#CED3D9';
            ctx.font = '10px "Inter", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(memory.label, x, y + orbSize + 15);
        });
        
        // Draw title
        ctx.fillStyle = '#CC785C';
        ctx.font = 'bold 16px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('F.R.E.D.\'s Memory System', centerX, 30);
        
        ctx.fillStyle = '#9DA7B3';
        ctx.font = '12px "JetBrains Mono", monospace';
        ctx.fillText('2D Fallback Visualization', centerX, 50);
        
        animationTime++;
        requestAnimationFrame(animate);
    }
    
    // Handle canvas resizing
    window.addEventListener('resize', () => {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    });
    
    container.innerHTML = '';
    container.appendChild(canvas);
    animate();
    
    // Add simple click interaction
    canvas.addEventListener('click', (event) => {
        const rect = canvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;
        
        // Check if clicked on core
        const distToCore = Math.sqrt((clickX - centerX) ** 2 + (clickY - centerY) ** 2);
        if (distToCore < 30) {
            alert('F.R.E.D. Central Processing Core\n\nThe heart of F.R.E.D.\'s consciousness and memory system.');
        }
    });
}
