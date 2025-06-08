/**
 * keywords: [3d, viewport, three.js, webgl, trajectory, visualization]
 * 
 * High-performance 3D viewport for agent trajectory visualization
 */

export class Viewport3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        this.data = null;
        this.currentTime = 0;
        
        // Visual elements
        this.agent = null;
        this.trajectory = null;
        this.rewards = [];
        this.heatmap = null;
        this.grid = null;
        
        // Performance optimization
        this.frameSkip = 0;
        this.LOD = {
            high: 0,
            medium: 1000,
            low: 5000
        };
        
        this.init();
    }
    
    init() {
        // Setup Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
        
        // Setup camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 100);
        this.camera.position.set(15, 20, 15);
        this.camera.lookAt(5, 0, 5);
        
        // Setup renderer with optimization
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            powerPreference: "high-performance"
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = false; // Disable for performance
        
        this.container.appendChild(this.renderer.domElement);
        
        // Setup controls
        this.setupControls();
        
        // Setup lights
        this.setupLights();
        
        // Setup grid
        this.setupGrid();
        
        // Handle resize
        window.addEventListener('resize', () => this.onResize());
        
        // Start render loop
        this.animate();
    }
    
    setupControls() {
        // Simple orbit controls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            mouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (!mouseDown) return;
            
            const deltaX = e.clientX - mouseX;
            const deltaY = e.clientY - mouseY;
            
            // Rotate camera around center
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(this.camera.position);
            spherical.theta -= deltaX * 0.01;
            spherical.phi += deltaY * 0.01;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            
            this.camera.position.setFromSpherical(spherical);
            this.camera.lookAt(5, 0, 5);
            
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            mouseDown = false;
        });
        
        // Zoom
        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            const scale = e.deltaY > 0 ? 1.1 : 0.9;
            this.camera.position.multiplyScalar(scale);
        });
    }
    
    setupLights() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);
        
        // Directional light
        const directional = new THREE.DirectionalLight(0xffffff, 0.8);
        directional.position.set(10, 20, 5);
        this.scene.add(directional);
        
        // Point light following agent
        this.agentLight = new THREE.PointLight(0x4a9eff, 0.5, 10);
        this.scene.add(this.agentLight);
    }
    
    setupGrid() {
        // Create efficient grid using instanced mesh
        const gridSize = 10;
        const cellSize = 1;
        
        // Grid lines
        const gridGeometry = new THREE.BufferGeometry();
        const gridPositions = [];
        const gridColors = [];
        
        // Create grid lines
        for (let i = 0; i <= gridSize; i++) {
            const color = i % 5 === 0 ? [0.4, 0.4, 0.4] : [0.2, 0.2, 0.2];
            
            // Horizontal lines
            gridPositions.push(0, 0, i);
            gridPositions.push(gridSize, 0, i);
            gridColors.push(...color, ...color);
            
            // Vertical lines
            gridPositions.push(i, 0, 0);
            gridPositions.push(i, 0, gridSize);
            gridColors.push(...color, ...color);
        }
        
        gridGeometry.setAttribute('position', 
            new THREE.Float32BufferAttribute(gridPositions, 3));
        gridGeometry.setAttribute('color', 
            new THREE.Float32BufferAttribute(gridColors, 3));
            
        const gridMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 1
        });
        
        this.grid = new THREE.LineSegments(gridGeometry, gridMaterial);
        this.scene.add(this.grid);
        
        // Grid plane for raycasting
        const planeGeometry = new THREE.PlaneGeometry(gridSize, gridSize);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0x1a1a1a,
            transparent: true,
            opacity: 0.1
        });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        plane.rotation.x = -Math.PI / 2;
        plane.position.set(gridSize / 2, -0.01, gridSize / 2);
        this.scene.add(plane);
    }
    
    setData(data) {
        this.data = data;
        this.currentTime = 0;
        
        // Clear existing objects
        this.clearVisualization();
        
        // Create agent
        this.createAgent();
        
        // Create trajectory trail
        this.createTrajectory();
        
        // Create rewards
        this.createRewards();
        
        // Create heatmap
        this.createHeatmap();
    }
    
    createAgent() {
        // Agent body (sphere)
        const bodyGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const bodyMaterial = new THREE.MeshPhongMaterial({
            color: 0x4a9eff,
            emissive: 0x4a9eff,
            emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        
        // Direction indicator (cone)
        const coneGeometry = new THREE.ConeGeometry(0.2, 0.4, 8);
        const coneMaterial = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            emissive: 0xffffff,
            emissiveIntensity: 0.1
        });
        const cone = new THREE.Mesh(coneGeometry, coneMaterial);
        cone.position.z = 0.3;
        cone.rotation.x = Math.PI / 2;
        
        // Group
        this.agent = new THREE.Group();
        this.agent.add(body);
        this.agent.add(cone);
        this.scene.add(this.agent);
        
        // Neural activity indicator (pulsing glow)
        const glowGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x4a9eff,
            transparent: true,
            opacity: 0.2
        });
        this.agentGlow = new THREE.Mesh(glowGeometry, glowMaterial);
        this.agent.add(this.agentGlow);
    }
    
    createTrajectory() {
        // Use line geometry for efficient trajectory rendering
        const maxPoints = Math.min(this.data.trajectory.x.length, 10000);
        const geometry = new THREE.BufferGeometry();
        
        // Pre-allocate buffer
        const positions = new Float32Array(maxPoints * 3);
        const colors = new Float32Array(maxPoints * 3);
        
        geometry.setAttribute('position', 
            new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', 
            new THREE.BufferAttribute(colors, 3));
            
        // Material with vertex colors
        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 2,
            transparent: true,
            opacity: 0.8
        });
        
        this.trajectory = new THREE.Line(geometry, material);
        this.trajectory.frustumCulled = false;
        this.scene.add(this.trajectory);
        
        // Trail end marker
        const markerGeometry = new THREE.SphereGeometry(0.1, 8, 8);
        const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        this.trailMarker = new THREE.Mesh(markerGeometry, markerMaterial);
        this.scene.add(this.trailMarker);
    }
    
    createRewards() {
        // Get unique reward positions
        const rewardPositions = new Map();
        
        this.data.trajectory.x.forEach((x, i) => {
            if (this.data.rewards[i] > 0) {
                const key = `${Math.floor(x)},${Math.floor(this.data.trajectory.y[i])}`;
                if (!rewardPositions.has(key)) {
                    rewardPositions.set(key, {
                        x: Math.floor(x) + 0.5,
                        y: Math.floor(this.data.trajectory.y[i]) + 0.5,
                        firstCollection: i
                    });
                }
            }
        });
        
        // Create reward meshes
        const rewardGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.1, 16);
        const rewardMaterial = new THREE.MeshPhongMaterial({
            color: 0xffeb3b,
            emissive: 0xffeb3b,
            emissiveIntensity: 0.3
        });
        
        rewardPositions.forEach((pos, key) => {
            const reward = new THREE.Mesh(rewardGeometry, rewardMaterial.clone());
            reward.position.set(pos.x, 0.05, pos.y);
            reward.userData = { collected: false, collectionTime: pos.firstCollection };
            this.rewards.push(reward);
            this.scene.add(reward);
        });
    }
    
    createHeatmap() {
        // Create visitation heatmap
        const gridSize = 10;
        const heatmapData = new Array(gridSize * gridSize).fill(0);
        
        // Count visits
        this.data.trajectory.x.forEach((x, i) => {
            const gridX = Math.floor(x);
            const gridY = Math.floor(this.data.trajectory.y[i]);
            if (gridX >= 0 && gridX < gridSize && gridY >= 0 && gridY < gridSize) {
                heatmapData[gridY * gridSize + gridX]++;
            }
        });
        
        // Normalize
        const maxVisits = Math.max(...heatmapData);
        
        // Create heatmap mesh
        const geometry = new THREE.PlaneGeometry(gridSize, gridSize, gridSize - 1, gridSize - 1);
        const vertices = geometry.attributes.position;
        
        // Apply height based on visits
        for (let i = 0; i < vertices.count; i++) {
            const x = Math.floor(vertices.getX(i) + gridSize / 2);
            const z = Math.floor(vertices.getZ(i) + gridSize / 2);
            const visits = heatmapData[z * gridSize + x] || 0;
            vertices.setY(i, (visits / maxVisits) * 0.5);
        }
        
        // Color based on visits
        const colors = new Float32Array(vertices.count * 3);
        for (let i = 0; i < vertices.count; i++) {
            const x = Math.floor(vertices.getX(i) + gridSize / 2);
            const z = Math.floor(vertices.getZ(i) + gridSize / 2);
            const visits = heatmapData[z * gridSize + x] || 0;
            const intensity = visits / maxVisits;
            
            // Color gradient from blue to red
            colors[i * 3] = intensity;
            colors[i * 3 + 1] = 0.2;
            colors[i * 3 + 2] = 1 - intensity;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        
        this.heatmap = new THREE.Mesh(geometry, material);
        this.heatmap.rotation.x = -Math.PI / 2;
        this.heatmap.position.set(gridSize / 2, -0.05, gridSize / 2);
        this.heatmap.visible = false; // Hidden by default
        this.scene.add(this.heatmap);
    }
    
    update(time) {
        if (!this.data) return;
        
        this.currentTime = time;
        const dataLength = this.data.trajectory.x.length;
        
        // Clamp time
        time = Math.max(0, Math.min(time, dataLength - 1));
        
        // Update agent position
        const x = this.data.trajectory.x[time];
        const y = this.data.trajectory.y[time];
        this.agent.position.set(x, 0.3, y);
        
        // Update agent rotation based on movement direction
        if (time > 0) {
            const prevX = this.data.trajectory.x[time - 1];
            const prevY = this.data.trajectory.y[time - 1];
            const dx = x - prevX;
            const dy = y - prevY;
            
            if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
                this.agent.rotation.y = Math.atan2(dx, dy);
            }
        }
        
        // Update agent glow based on neural activity
        const neuralActivity = this.calculateNeuralActivity(time);
        this.agentGlow.scale.setScalar(1 + neuralActivity * 0.5);
        this.agentGlow.material.opacity = 0.1 + neuralActivity * 0.3;
        
        // Update light position
        this.agentLight.position.copy(this.agent.position);
        this.agentLight.position.y = 2;
        
        // Update trajectory
        this.updateTrajectory(time);
        
        // Update rewards
        this.updateRewards(time);
        
        // Update trail marker
        if (time > 0) {
            const trailEnd = Math.max(0, time - 100);
            this.trailMarker.position.set(
                this.data.trajectory.x[trailEnd],
                0.1,
                this.data.trajectory.y[trailEnd]
            );
        }
    }
    
    updateTrajectory(currentTime) {
        const positions = this.trajectory.geometry.attributes.position;
        const colors = this.trajectory.geometry.attributes.color;
        
        // Determine visible range (show last N points)
        const trailLength = 1000;
        const startTime = Math.max(0, currentTime - trailLength);
        const visiblePoints = currentTime - startTime;
        
        // Update positions and colors
        for (let i = 0; i < visiblePoints; i++) {
            const t = startTime + i;
            
            positions.setXYZ(i, 
                this.data.trajectory.x[t],
                0.1,
                this.data.trajectory.y[t]
            );
            
            // Color based on value or reward
            const value = this.data.values ? this.data.values[t] : 0;
            const reward = this.data.rewards[t];
            
            if (reward > 0) {
                colors.setXYZ(i, 1, 0.9, 0); // Yellow for rewards
            } else {
                const intensity = Math.max(0, Math.min(1, value));
                colors.setXYZ(i, 
                    intensity,
                    0.5 + intensity * 0.5,
                    1 - intensity
                );
            }
        }
        
        // Hide unused points
        for (let i = visiblePoints; i < positions.count; i++) {
            positions.setXYZ(i, 0, -100, 0);
        }
        
        positions.needsUpdate = true;
        colors.needsUpdate = true;
        
        // Update geometry bounding sphere for culling
        this.trajectory.geometry.computeBoundingSphere();
    }
    
    updateRewards(time) {
        // Check reward collection
        const x = this.data.trajectory.x[time];
        const y = this.data.trajectory.y[time];
        
        this.rewards.forEach(reward => {
            if (!reward.userData.collected && time >= reward.userData.collectionTime) {
                // Animate collection
                reward.userData.collected = true;
                reward.material.color.setHex(0x00ff00);
                reward.material.emissiveIntensity = 1;
                
                // Fade out
                const startScale = reward.scale.y;
                const animate = () => {
                    reward.scale.y *= 0.95;
                    reward.material.opacity *= 0.95;
                    
                    if (reward.scale.y > 0.01) {
                        requestAnimationFrame(animate);
                    } else {
                        reward.visible = false;
                    }
                };
                animate();
            }
        });
    }
    
    calculateNeuralActivity(time) {
        // Calculate normalized neural activity for visual feedback
        if (!this.data.neural || !this.data.neural.spikes) return 0;
        
        // Simple spike count in a window
        const window = 50; // 50ms window
        const start = Math.max(0, time - window);
        let spikeCount = 0;
        
        for (let t = start; t < time && t < this.data.neural.spikes.length; t++) {
            if (this.data.neural.spikes[t]) {
                spikeCount += this.data.neural.spikes[t].reduce((a, b) => a + b, 0);
            }
        }
        
        // Normalize (assuming ~1000 neurons, ~10Hz average rate)
        return Math.min(1, spikeCount / (1000 * 0.01 * window));
    }
    
    toggleHeatmap() {
        this.heatmap.visible = !this.heatmap.visible;
    }
    
    resetCamera() {
        this.camera.position.set(15, 20, 15);
        this.camera.lookAt(5, 0, 5);
    }
    
    clearVisualization() {
        // Remove agent
        if (this.agent) {
            this.scene.remove(this.agent);
            this.agent = null;
        }
        
        // Remove trajectory
        if (this.trajectory) {
            this.trajectory.geometry.dispose();
            this.trajectory.material.dispose();
            this.scene.remove(this.trajectory);
            this.trajectory = null;
        }
        
        // Remove rewards
        this.rewards.forEach(reward => {
            reward.geometry.dispose();
            reward.material.dispose();
            this.scene.remove(reward);
        });
        this.rewards = [];
        
        // Remove heatmap
        if (this.heatmap) {
            this.heatmap.geometry.dispose();
            this.heatmap.material.dispose();
            this.scene.remove(this.heatmap);
            this.heatmap = null;
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Skip frames for performance if needed
        this.frameSkip++;
        if (this.frameSkip % 2 === 0) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        // Clean up resources
        this.clearVisualization();
        this.renderer.dispose();
        this.scene.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }
}