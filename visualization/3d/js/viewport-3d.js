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
        this.gridSize = [10, 10]; // Default grid size
        
        // Camera controls
        this.followAgent = false;
        this.showHeatmap = false;
        
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
        // No fog - we want to see everything from any distance
        
        // Setup camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        // Initial position - will be updated when data is loaded
        this.camera.position.set(20, 15, 5);
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
        
        // Setup camera control checkboxes
        this.setupCameraControls();
        
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
            this.camera.lookAt(this.gridSize[0] / 2, 0, this.gridSize[1] / 2);
            
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
    
    setupCameraControls() {
        const followCheckbox = document.getElementById('camera-follow');
        const heatmapCheckbox = document.getElementById('show-heatmap');
        
        if (followCheckbox) {
            followCheckbox.addEventListener('change', (e) => {
                this.followAgent = e.target.checked;
                if (!this.followAgent) {
                    // Reset camera when turning off follow mode
                    const centerX = this.gridSize[0] / 2;
                    const centerZ = this.gridSize[1] / 2;
                    const distance = Math.max(this.gridSize[0], this.gridSize[1]) * 1.5;
                    this.camera.position.set(centerX + distance, distance * 0.7, centerZ);
                    this.camera.lookAt(centerX, 0, centerZ);
                }
            });
        }
        
        if (heatmapCheckbox) {
            heatmapCheckbox.addEventListener('change', (e) => {
                this.showHeatmap = e.target.checked;
                if (this.heatmap) {
                    this.heatmap.visible = this.showHeatmap;
                }
            });
        }
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
        // Clear existing grid if any
        if (this.grid) {
            this.scene.remove(this.grid);
            this.grid.geometry.dispose();
            this.grid.material.dispose();
        }
        
        // Create efficient grid using instanced mesh
        const gridSizeX = this.gridSize[0];
        const gridSizeZ = this.gridSize[1];
        const cellSize = 1;
        
        // Grid lines
        const gridGeometry = new THREE.BufferGeometry();
        const gridPositions = [];
        const gridColors = [];
        
        // Create grid lines
        for (let i = 0; i <= gridSizeX; i++) {
            const color = i % 5 === 0 ? [0.4, 0.4, 0.4] : [0.2, 0.2, 0.2];
            
            // Lines along Z axis
            gridPositions.push(i, 0, 0);
            gridPositions.push(i, 0, gridSizeZ);
            gridColors.push(...color, ...color);
        }
        
        for (let i = 0; i <= gridSizeZ; i++) {
            const color = i % 5 === 0 ? [0.4, 0.4, 0.4] : [0.2, 0.2, 0.2];
            
            // Lines along X axis
            gridPositions.push(0, 0, i);
            gridPositions.push(gridSizeX, 0, i);
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
        const planeGeometry = new THREE.PlaneGeometry(gridSizeX, gridSizeZ);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0x1a1a1a,
            transparent: true,
            opacity: 0.1
        });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        plane.rotation.x = -Math.PI / 2;
        plane.position.set(gridSizeX / 2, -0.01, gridSizeZ / 2);
        this.scene.add(plane);
        
        // Also update camera to look at the center of the grid
        this.camera.lookAt(gridSizeX / 2, 0, gridSizeZ / 2);
    }
    
    setData(data) {
        this.data = data;
        this.currentTime = 0;
        
        // Extract grid size from metadata if available
        console.log('Viewport3D setData - metadata:', data.metadata);
        if (data.metadata && data.metadata.grid_size) {
            this.gridSize = data.metadata.grid_size;
            console.log('Setting grid size to:', this.gridSize);
            // Recreate grid with new size
            this.setupGrid();
            // Update camera position for new grid - view from side
            const centerX = this.gridSize[0] / 2;
            const centerZ = this.gridSize[1] / 2;
            const distance = Math.max(this.gridSize[0], this.gridSize[1]) * 1.5;
            // Position camera to view from the side (along X axis)
            this.camera.position.set(centerX + distance, distance * 0.7, centerZ);
            this.camera.lookAt(centerX, 0, centerZ);
        }
        
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
        // Instead of a single line, use line segments to handle toroidal wrapping
        const material = new THREE.LineBasicMaterial({
            color: 0x4a9eff,
            linewidth: 2,
            transparent: true,
            opacity: 0.8
        });
        
        // Create a group to hold trajectory segments
        this.trajectoryGroup = new THREE.Group();
        this.scene.add(this.trajectoryGroup);
        
        // We'll create segments dynamically in updateTrajectory
        this.trajectorySegments = [];
        
        // Trail end marker
        const markerGeometry = new THREE.SphereGeometry(0.1, 8, 8);
        const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        this.trailMarker = new THREE.Mesh(markerGeometry, markerMaterial);
        this.trailMarker.visible = false; // Start hidden
        this.scene.add(this.trailMarker);
    }
    
    isWrapping(x1, y1, x2, y2) {
        // Detect if movement wrapped around the grid
        const dx = Math.abs(x2 - x1);
        const dy = Math.abs(y2 - y1);
        const threshold = Math.min(this.gridSize[0], this.gridSize[1]) * 0.5;
        return dx > threshold || dy > threshold;
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
            let dx = x - prevX;
            let dy = y - prevY;
            
            // Check for wrapping
            if (Math.abs(dx) > this.gridSize[0] * 0.5) {
                dx = dx > 0 ? dx - this.gridSize[0] : dx + this.gridSize[0];
            }
            if (Math.abs(dy) > this.gridSize[1] * 0.5) {
                dy = dy > 0 ? dy - this.gridSize[1] : dy + this.gridSize[1];
            }
            
            if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
                this.agent.rotation.y = Math.atan2(dx, dy);
            }
        }
        
        // Update camera if following
        if (this.followAgent && this.agent) {
            const offset = 10;
            this.camera.position.set(
                this.agent.position.x + offset,
                offset,
                this.agent.position.z + offset
            );
            this.camera.lookAt(this.agent.position);
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
        if (time > 100) {
            this.trailMarker.visible = true;
            const trailEnd = Math.max(0, time - 100);
            this.trailMarker.position.set(
                this.data.trajectory.x[trailEnd],
                0.1,
                this.data.trajectory.y[trailEnd]
            );
        } else {
            this.trailMarker.visible = false;
        }
    }
    
    updateTrajectory(currentTime) {
        // Clear existing segments
        this.trajectoryGroup.children.forEach(child => {
            child.geometry.dispose();
        });
        this.trajectoryGroup.clear();
        
        // Determine visible range (show last N points)
        const trailLength = 1000;
        const startTime = Math.max(0, currentTime - trailLength);
        
        if (startTime >= currentTime) return;
        
        // Build segments, breaking at wrap points
        let segmentPoints = [];
        let lastX = this.data.trajectory.x[startTime];
        let lastY = this.data.trajectory.y[startTime];
        
        for (let t = startTime; t <= currentTime && t < this.data.trajectory.x.length; t++) {
            const x = this.data.trajectory.x[t];
            const y = this.data.trajectory.y[t];
            
            // Check for wrapping
            if (t > startTime && this.isWrapping(lastX, lastY, x, y)) {
                // Create segment with current points
                if (segmentPoints.length > 1) {
                    this.createTrajectorySegment(segmentPoints);
                }
                segmentPoints = [];
            }
            
            segmentPoints.push(new THREE.Vector3(x, 0.1, y));
            lastX = x;
            lastY = y;
        }
        
        // Create final segment
        if (segmentPoints.length > 1) {
            this.createTrajectorySegment(segmentPoints);
        }
    }
    
    createTrajectorySegment(points) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x4a9eff,
            linewidth: 2,
            transparent: true,
            opacity: 0.8
        });
        const line = new THREE.Line(geometry, material);
        this.trajectoryGroup.add(line);
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
        
        // Remove trajectory segments
        if (this.trajectoryGroup) {
            this.trajectoryGroup.children.forEach(child => {
                child.geometry.dispose();
                child.material.dispose();
            });
            this.scene.remove(this.trajectoryGroup);
            this.trajectoryGroup = null;
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