/**
 * F.R.E.D. Neural Network Background Animation
 * Creates a subtle animated neural network in the background
 */

class NeuralNetwork {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.mousePos = { x: 0, y: 0 };
        this.animationId = null;
        
        this.config = {
            nodeCount: 15,
            maxDistance: 150,
            nodeSpeed: 0.3,
            nodeRadius: 2,
            connectionOpacity: 0.15,
            nodeOpacity: 0.3,
            pulseSpeed: 0.02,
            interactionRadius: 100
        };
        
        this.init();
    }
    
    init() {
        this.resize();
        this.createNodes();
        this.setupEventListeners();
        this.animate();
    }
    
    resize() {
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = window.innerWidth * dpr;
        this.canvas.height = window.innerHeight * dpr;
        this.canvas.style.width = window.innerWidth + 'px';
        this.canvas.style.height = window.innerHeight + 'px';
        this.ctx.scale(dpr, dpr);
        
        this.width = window.innerWidth;
        this.height = window.innerHeight;
    }
    
    createNodes() {
        this.nodes = [];
        for (let i = 0; i < this.config.nodeCount; i++) {
            this.nodes.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * this.config.nodeSpeed,
                vy: (Math.random() - 0.5) * this.config.nodeSpeed,
                pulse: Math.random() * Math.PI * 2,
                baseRadius: this.config.nodeRadius + Math.random() * 2
            });
        }
    }
    
    setupEventListeners() {
        window.addEventListener('resize', () => this.resize());
        
        document.addEventListener('mousemove', (e) => {
            this.mousePos.x = e.clientX;
            this.mousePos.y = e.clientY;
        });
    }
    
    updateNodes() {
        this.nodes.forEach(node => {
            // Update position
            node.x += node.vx;
            node.y += node.vy;
            
            // Bounce off edges
            if (node.x < 0 || node.x > this.width) node.vx *= -1;
            if (node.y < 0 || node.y > this.height) node.vy *= -1;
            
            // Keep nodes in bounds
            node.x = Math.max(0, Math.min(this.width, node.x));
            node.y = Math.max(0, Math.min(this.height, node.y));
            
            // Update pulse
            node.pulse += this.config.pulseSpeed;
            
            // Mouse interaction
            const mouseDistance = Math.sqrt(
                Math.pow(node.x - this.mousePos.x, 2) + 
                Math.pow(node.y - this.mousePos.y, 2)
            );
            
            if (mouseDistance < this.config.interactionRadius) {
                const force = (this.config.interactionRadius - mouseDistance) / this.config.interactionRadius;
                const angle = Math.atan2(node.y - this.mousePos.y, node.x - this.mousePos.x);
                node.vx += Math.cos(angle) * force * 0.1;
                node.vy += Math.sin(angle) * force * 0.1;
                
                // Limit velocity
                const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
                if (speed > this.config.nodeSpeed * 2) {
                    node.vx = (node.vx / speed) * this.config.nodeSpeed * 2;
                    node.vy = (node.vy / speed) * this.config.nodeSpeed * 2;
                }
            }
        });
    }
    
    drawConnections() {
        this.connections = [];
        
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const distance = Math.sqrt(
                    Math.pow(this.nodes[i].x - this.nodes[j].x, 2) + 
                    Math.pow(this.nodes[i].y - this.nodes[j].y, 2)
                );
                
                if (distance < this.config.maxDistance) {
                    const opacity = (1 - distance / this.config.maxDistance) * this.config.connectionOpacity;
                    
                    this.ctx.strokeStyle = `rgba(97, 170, 242, ${opacity})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.nodes[i].x, this.nodes[i].y);
                    this.ctx.lineTo(this.nodes[j].x, this.nodes[j].y);
                    this.ctx.stroke();
                    
                    this.connections.push({
                        from: i,
                        to: j,
                        distance: distance,
                        opacity: opacity
                    });
                }
            }
        }
    }
    
    drawNodes() {
        this.nodes.forEach(node => {
            const pulseRadius = node.baseRadius + Math.sin(node.pulse) * 0.5;
            const pulseOpacity = this.config.nodeOpacity + Math.sin(node.pulse) * 0.1;
            
            // Node glow
            this.ctx.shadowColor = '#61AAF2';
            this.ctx.shadowBlur = 10;
            
            // Main node
            this.ctx.fillStyle = `rgba(97, 170, 242, ${pulseOpacity})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, pulseRadius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Inner core
            this.ctx.shadowBlur = 5;
            this.ctx.fillStyle = `rgba(255, 255, 255, ${pulseOpacity * 0.8})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, pulseRadius * 0.4, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Reset shadow
            this.ctx.shadowBlur = 0;
        });
    }
    
    drawDataFlow() {
        const time = Date.now() * 0.001;
        
        this.connections.forEach((connection, index) => {
            if (Math.random() < 0.02) { // Random data flow
                const fromNode = this.nodes[connection.from];
                const toNode = this.nodes[connection.to];
                
                const progress = (Math.sin(time * 2 + index) + 1) / 2;
                const x = fromNode.x + (toNode.x - fromNode.x) * progress;
                const y = fromNode.y + (toNode.y - fromNode.y) * progress;
                
                this.ctx.fillStyle = `rgba(204, 120, 92, ${connection.opacity * 2})`;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 1.5, 0, Math.PI * 2);
                this.ctx.fill();
            }
        });
    }
    
    animate() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        this.updateNodes();
        this.drawConnections();
        this.drawNodes();
        this.drawDataFlow();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        window.removeEventListener('resize', this.resize);
    }
}

// Initialize neural network when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('neural-canvas');
    if (canvas) {
        window.neuralNetwork = new NeuralNetwork(canvas);
        
        // Add activity pulse when F.R.E.D. is speaking or processing
        window.neuralPulse = function(intensity = 1) {
            if (window.neuralNetwork) {
                window.neuralNetwork.config.pulseSpeed = 0.02 * (1 + intensity);
                window.neuralNetwork.config.connectionOpacity = 0.15 * (1 + intensity * 0.5);
                
                // Reset after a delay
                setTimeout(() => {
                    if (window.neuralNetwork) {
                        window.neuralNetwork.config.pulseSpeed = 0.02;
                        window.neuralNetwork.config.connectionOpacity = 0.15;
                    }
                }, 2000);
            }
        };
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.neuralNetwork) {
        window.neuralNetwork.destroy();
    }
}); 