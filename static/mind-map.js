document.addEventListener('DOMContentLoaded', async () => {
    const svg = d3.select("svg#mindMapVisualization");
    const width = +svg.attr("width") || 960;
    const height = +svg.attr("height") || 700;
    
    // Clear any previous content
    svg.selectAll("*").remove();
    
    // Add a proper viewBox for responsive scaling
    svg.attr("viewBox", [0, 0, width, height]);
    
    // Create a background for better visibility
    svg.append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "none");
    
    // Add loading message
    const loadingMessage = svg.append("text")
        .attr("x", width / 2)
        .attr("y", height / 2)
        .attr("text-anchor", "middle")
        .attr("font-family", "sans-serif")
        .attr("font-size", "20px")
        .text("Loading F.R.E.D.'s Memories...");
    
    try {
        // Fetch memory data from API endpoint
        const response = await fetch('/api/memory_visualization_data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        let memoryNodes = await response.json();
        loadingMessage.remove();
        
        if (!memoryNodes || memoryNodes.length === 0) {
            svg.append("text")
                .attr("x", width / 2)
                .attr("y", height / 2)
                .attr("text-anchor", "middle")
                .text("No memories found to visualize.");
            return;
        }
        
        // --- 1. Create Brain Node ---
        const centralBrainNode = {
            nodeid: "FRED_BRAIN",
            id: "FRED_BRAIN",
            label: "F.R.E.D.'s Brain",
            type: "Brain",
            embedding: null,
            total_edge_count: Math.max(...memoryNodes.map(n => n.total_edge_count)) * 2,
            isBrain: true
        };
        
        // --- 2. Calculate node relationships and hierarchy ---
        function cosineSimilarity(vecA, vecB) {
            if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0 || vecA.length !== vecB.length) {
                return 0;
            }
            let dotProduct = 0;
            let normA = 0;
            let normB = 0;
            for (let i = 0; i < vecA.length; i++) {
                dotProduct += vecA[i] * vecB[i];
                normA += vecA[i] * vecA[i];
                normB += vecB[i] * vecB[i];
            }
            normA = Math.sqrt(normA);
            normB = Math.sqrt(normB);
            if (normA === 0 || normB === 0) return 0;
            return dotProduct / (normA * normB);
        }
        
        // Add id property to all memory nodes for D3
        memoryNodes = memoryNodes.map(node => ({
            ...node,
            id: node.nodeid.toString(),
            satellites: [], // Will store nodes that orbit this node
            parentNode: null // Will reference the node this orbits (if not brain)
        }));
        
        // Sort nodes by importance (total_edge_count)
        const sortedNodes = [...memoryNodes].sort((a, b) => b.total_edge_count - a.total_edge_count);
        
        // --- 3. Create hierarchical orbital structure ---
        
        // Identify primary planets (orbiting the brain directly)
        const primaryPlanets = sortedNodes.slice(0, 6); // Top 6 nodes orbit brain directly
        const secondaryNodes = sortedNodes.slice(6); // Rest are available to be satellites
        
        // Assign primary planets to orbital rings around the brain
        const brainOrbitalRings = [
            { radius: 150, nodes: [] },
            { radius: 230, nodes: [] },
            { radius: 310, nodes: [] }
        ];
        
        // Distribute primary planets across the orbital rings
        primaryPlanets.forEach((node, i) => {
            const ringIndex = Math.min(i % brainOrbitalRings.length, brainOrbitalRings.length - 1);
            const ring = brainOrbitalRings[ringIndex];
            
            // Assign orbit data
            node.orbitRadius = ring.radius;
            node.orbitingBrain = true;
            node.parentNode = centralBrainNode;
            
            // Calculate initial position on orbit
            const angleStep = (2 * Math.PI) / Math.max(3, primaryPlanets.length / brainOrbitalRings.length);
            node.orbitAngle = (ring.nodes.length) * angleStep;
            node.x = width/2 + node.orbitRadius * Math.cos(node.orbitAngle);
            node.y = height/2 + node.orbitRadius * Math.sin(node.orbitAngle);
            
            // Add to ring collection
            ring.nodes.push(node);
            
            // Track orbital speed - inner orbits move faster
            node.orbitSpeed = 0.0008 * (1 + (1 - ringIndex * 0.2));
        });
        
        // Create satellite assignments - Group similar nodes with their parent planets
        secondaryNodes.forEach(node => {
            // Find most similar primary planet to orbit based on embedding similarity
            let bestParent = null;
            let highestSimilarity = 0.1; // Minimum threshold for similarity
            
            for (const planet of primaryPlanets) {
                if (node.embedding && planet.embedding) {
                    const similarity = cosineSimilarity(node.embedding, planet.embedding);
                    if (similarity > highestSimilarity) {
                        highestSimilarity = similarity;
                        bestParent = planet;
                    }
                }
            }
            
            // If we found a similar enough planet, make this node its satellite
            if (bestParent) {
                node.parentNode = bestParent;
                bestParent.satellites.push(node);
                node.orbitingBrain = false;
                
                // Calculate orbital data relative to parent planet
                // Satellites orbit in a smaller radius around their parent
                const satelliteCount = bestParent.satellites.length;
                node.satelliteIndex = satelliteCount - 1;
                
                // Size satellite orbit based on parent's importance
                const minOrbitSize = 40;
                const maxOrbitSize = 90;
                const orbitSizeRange = maxOrbitSize - minOrbitSize;
                const parentSizeRatio = bestParent.total_edge_count / primaryPlanets[0].total_edge_count;
                node.orbitRadius = minOrbitSize + (orbitSizeRange * parentSizeRatio);
                
                // Initial position and angle on the orbit
                const angleStep = (2 * Math.PI) / Math.max(1, satelliteCount);
                node.orbitAngle = node.satelliteIndex * angleStep;
                
                // Satellites orbit faster than their parent planets
                node.orbitSpeed = bestParent.orbitSpeed * 2.2;
            } else {
                // If no similar planet, orbit the brain in outer rings
                const outerRings = [
                    { radius: 390, nodes: [] },
                    { radius: 460, nodes: [] }
                ];
                
                // Assign to outer rings
                const ringIndex = node.satelliteIndex % 2; // Alternate between the two outer rings
                const ring = outerRings[ringIndex];
                
                node.orbitRadius = ring.radius;
                node.orbitingBrain = true;
                node.parentNode = centralBrainNode;
                
                // Calculate position
                const angleStep = (2 * Math.PI) / Math.max(8, secondaryNodes.length / 2);
                node.orbitAngle = (ring.nodes.length) * angleStep;
                node.x = width/2 + node.orbitRadius * Math.cos(node.orbitAngle);
                node.y = height/2 + node.orbitRadius * Math.sin(node.orbitAngle);
                
                // Outer orbits move slower
                node.orbitSpeed = 0.0003;
                
                // Add to ring
                ring.nodes.push(node);
            }
        });
        
        // Set brain node position at center
        centralBrainNode.x = width/2;
        centralBrainNode.y = height/2;
        centralBrainNode.fx = width/2; // Fix position X
        centralBrainNode.fy = height/2; // Fix position Y
        
        // Create links for visualization
        const links = [];
        
        // Add links from primary planets to brain
        primaryPlanets.forEach(planet => {
            links.push({
                source: planet.id,
                target: centralBrainNode.id,
                value: 0.8,
                isPrimary: true,
                isVisible: true
            });
        });
        
        // Add links from satellites to their parent planets
        memoryNodes.forEach(node => {
            if (node.parentNode && !node.orbitingBrain) {
                links.push({
                    source: node.id,
                    target: node.parentNode.id,
                    value: 0.6,
                    isSatellite: true,
                    isVisible: true
                });
            }
        });
        
        // Add a few cross-connections between similar nodes for visual interest
        // But limit them to avoid spider-web appearance
        const maxCrossLinks = Math.min(10, Math.floor(memoryNodes.length / 5));
        let crossLinkCount = 0;
        
        for (let i = 0; i < memoryNodes.length && crossLinkCount < maxCrossLinks; i++) {
            // Only add cross-links with some randomness to avoid clutter
            if (Math.random() < 0.3) continue;
            
            const nodeA = memoryNodes[i];
            for (let j = i + 1; j < memoryNodes.length && crossLinkCount < maxCrossLinks; j++) {
                const nodeB = memoryNodes[j];
                
                // Don't link nodes that already have a parent-satellite relationship
                if (nodeA.parentNode === nodeB || nodeB.parentNode === nodeA) continue;
                
                // Only connect nodes of the same type and with embeddings
                if (nodeA.type === nodeB.type && nodeA.embedding && nodeB.embedding) {
                    const similarity = cosineSimilarity(nodeA.embedding, nodeB.embedding);
                    
                    // Only add highly similar connections
                    if (similarity > 0.7) {
                        links.push({
                            source: nodeA.id,
                            target: nodeB.id,
                            value: similarity * 0.5, // Thinner lines for cross-connections
                            isCrossLink: true,
                            isVisible: true
                        });
                        crossLinkCount++;
                    }
                }
            }
        }
        
        // Combine all nodes
        const allNodes = [centralBrainNode, ...memoryNodes];
        
        // --- 4. Set up node sizing based on importance ---
        const minEdgeCount = d3.min(allNodes, d => d.total_edge_count) || 1;
        const maxEdgeCount = d3.max(allNodes, d => d.total_edge_count) || 10;
        
        const nodeRadiusScale = d3.scaleSqrt()
            .domain([minEdgeCount, maxEdgeCount])
            .range([3, 18]);
        
        // --- 5. Create visual elements ---
        const defs = svg.append("defs");
        
        // Gradient for Brain node
        const brainGradient = defs.append("radialGradient")
            .attr("id", "brainGradient")
            .attr("cx", "50%")
            .attr("cy", "50%")
            .attr("r", "70%")
            .attr("fx", "50%")
            .attr("fy", "50%");
            
        brainGradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", "#00BFFF")
            .attr("stop-opacity", 1);
            
        brainGradient.append("stop")
            .attr("offset", "70%")
            .attr("stop-color", "#0080FF")
            .attr("stop-opacity", 1);
            
        brainGradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "#0040FF")
            .attr("stop-opacity", 1);
        
        // Create groups for visualization elements
        const orbitalGroup = svg.append("g").attr("class", "orbits");
        const linkGroup = svg.append("g").attr("class", "links");
        const nodeGroup = svg.append("g").attr("class", "nodes");
        
        // Create subtle brain orbital rings
        brainOrbitalRings.forEach(ring => {
            orbitalGroup.append("circle")
                .attr("cx", width/2)
                .attr("cy", height/2)
                .attr("r", ring.radius)
                .attr("fill", "none")
                .attr("stroke", "rgba(150, 150, 150, 0.1)")
                .attr("stroke-width", 1)
                .attr("stroke-dasharray", "3,3");
        });
        
        // Create links (paths)
        const link = linkGroup.selectAll("path")
            .data(links.filter(d => d.isVisible))
            .enter()
            .append("path")
            .attr("fill", "none")
            .attr("stroke", d => {
                if (d.isPrimary) return "#7ac7ff"; // Links to brain are blue
                if (d.isSatellite) return "#98e3b9"; // Satellite links are green-tinted
                if (d.isCrossLink) return "#d9c2ef"; // Cross links are purple-tinted
                return "#aaa"; // Fallback gray
            })
            .attr("stroke-opacity", d => d.isCrossLink ? 0.3 : 0.5)
            .attr("stroke-width", d => {
                if (d.isPrimary) return Math.max(1.5, Math.sqrt(d.value) * 2.5);
                if (d.isSatellite) return Math.max(1, Math.sqrt(d.value) * 2);
                return Math.max(0.5, d.value * 1.5); // Thinner for cross-links
            });
        
        // Create nodes
        const node = nodeGroup.selectAll("g")
            .data(allNodes)
            .enter()
            .append("g")
            .attr("class", d => {
                if (d.isBrain) return "brain-node";
                if (d.orbitingBrain) return "planet-node";
                return "satellite-node";
            });
        
        // Add circles for nodes
        node.append("circle")
            .attr("r", d => {
                if (d.isBrain) return 35;
                if (d.orbitingBrain) return nodeRadiusScale(d.total_edge_count) * 1.2;
                return nodeRadiusScale(d.total_edge_count) * 0.8;
            })
            .attr("fill", d => {
                if (d.isBrain) return "url(#brainGradient)";
                
                // Color by node type with better alpha for depth
                const alpha = d.orbitingBrain ? 0.9 : 0.8;
                switch (d.type) {
                    case 'Semantic': return `rgba(66, 133, 244, ${alpha})`; // Blue
                    case 'Episodic': return `rgba(234, 67, 53, ${alpha})`; // Red
                    case 'Procedural': return `rgba(52, 168, 83, ${alpha})`; // Green
                    default: return `rgba(251, 188, 5, ${alpha})`; // Yellow
                }
            })
            .attr("stroke", d => {
                if (d.isBrain) return "#fff";
                
                // Planets have brighter strokes than satellites
                const brightness = d.orbitingBrain ? 0.9 : 0.7;
                switch (d.type) {
                    case 'Semantic': return `rgba(120, 180, 255, ${brightness})`;
                    case 'Episodic': return `rgba(255, 130, 120, ${brightness})`;
                    case 'Procedural': return `rgba(100, 220, 130, ${brightness})`;
                    default: return `rgba(255, 220, 100, ${brightness})`;
                }
            })
            .attr("stroke-width", d => d.isBrain ? 2 : 1)
            .style("filter", d => {
                if (d.isBrain) return "drop-shadow(0 0 10px rgba(0, 191, 255, 0.5))";
                if (d.orbitingBrain) return "drop-shadow(0 0 5px rgba(255, 255, 255, 0.2))";
                return "";
            });
        
        // Add glow effect for nodes
        node.append("circle")
            .attr("r", d => {
                if (d.isBrain) return 42;
                if (d.orbitingBrain) return nodeRadiusScale(d.total_edge_count) * 1.2 + 5;
                return nodeRadiusScale(d.total_edge_count) * 0.8 + 3;
            })
            .attr("fill", "none")
            .attr("stroke", d => {
                if (d.isBrain) return "rgba(0, 191, 255, 0.2)";
                
                const alpha = d.orbitingBrain ? 0.2 : 0.15;
                switch (d.type) {
                    case 'Semantic': return `rgba(66, 133, 244, ${alpha})`;
                    case 'Episodic': return `rgba(234, 67, 53, ${alpha})`;
                    case 'Procedural': return `rgba(52, 168, 83, ${alpha})`;
                    default: return `rgba(251, 188, 5, ${alpha})`;
                }
            })
            .attr("stroke-width", d => d.isBrain ? 5 : (d.orbitingBrain ? 4 : 2));
        
        // Add labels to nodes (only for brain and primary planets)
        node.filter(d => d.isBrain || d.orbitingBrain)
            .append("text")
            .attr("dy", d => d.isBrain ? "0.5em" : "2.5em")
            .attr("text-anchor", "middle")
            .attr("font-size", d => {
                if (d.isBrain) return "14px";
                if (d.orbitingBrain) return "9px";
                return "8px";
            })
            .attr("font-family", "sans-serif")
            .attr("fill", "#f1f1f1")
            .attr("stroke", "rgba(0, 0, 0, 0.5)")
            .attr("stroke-width", "0.5px")
            .attr("pointer-events", "none")
            .text(d => {
                // Truncate long labels
                const maxLength = d.isBrain ? 20 : 12;
                return d.label.length > maxLength 
                    ? d.label.substring(0, maxLength - 3) + "..." 
                    : d.label;
            });
        
        // Add tooltips
        node.append("title")
            .text(d => {
                let tooltip = `${d.label}\nType: ${d.type}\nConnections: ${d.total_edge_count}`;
                if (d.orbitingBrain) {
                    tooltip += `\nSatellites: ${d.satellites.length}`;
                } else if (!d.isBrain) {
                    tooltip += `\nOrbiting: ${d.parentNode.label}`;
                }
                return tooltip;
            });
            
        // --- 6. Set up animation ---
        function updateOrbitalPositions() {
            // Update positions for planets orbiting brain
            allNodes.forEach(node => {
                if (node.isBrain) return; // Skip brain
                
                // Update orbit angle
                node.orbitAngle += node.orbitSpeed || 0.0005;
                
                if (node.orbitingBrain) {
                    // Primary planets orbit the brain
                    node.x = width/2 + node.orbitRadius * Math.cos(node.orbitAngle);
                    node.y = height/2 + node.orbitRadius * Math.sin(node.orbitAngle);
                } else if (node.parentNode) {
                    // Satellites orbit their parent planets
                    // The parent planet has already been positioned
                    node.x = node.parentNode.x + node.orbitRadius * Math.cos(node.orbitAngle);
                    node.y = node.parentNode.y + node.orbitRadius * Math.sin(node.orbitAngle);
                }
            });
            
            // Update link paths
            link.attr("d", d => {
                const source = allNodes.find(n => n.id === d.source);
                const target = allNodes.find(n => n.id === d.target);
                
                // Direct straight lines for brain connections
                if (source.isBrain || target.isBrain) {
                    return `M${source.x},${source.y} L${target.x},${target.y}`;
                }
                
                // Curved paths for other connections
                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const dr = Math.sqrt(dx * dx + dy * dy) * 1.2;
                return `M${source.x},${source.y} A${dr},${dr} 0 0,1 ${target.x},${target.y}`;
            });
            
            // Update node positions
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        }
        
        // Add pulsing animation to Brain node
        const brainNodeElement = node.filter(d => d.isBrain);
        const pulse = () => {
            brainNodeElement.select("circle:first-child")
                .transition()
                .duration(3000)
                .attr("r", 38)
                .attr("stroke-opacity", 0.8)
                .transition()
                .duration(3000)
                .attr("r", 35)
                .attr("stroke-opacity", 1)
                .on("end", pulse);
                
            brainNodeElement.select("circle:nth-child(2)")
                .transition()
                .duration(3000)
                .attr("r", 45)
                .transition()
                .duration(3000)
                .attr("r", 42)
                .on("end", pulse);
        };
        
        // Start animations
        pulse();
        
        // Use D3's timer for smooth animation
        d3.timer(updateOrbitalPositions);
        
    } catch (error) {
        console.error("Error creating mind map visualization:", error);
        loadingMessage.text(`Failed to load visualization: ${error.message}`);
    }
});
