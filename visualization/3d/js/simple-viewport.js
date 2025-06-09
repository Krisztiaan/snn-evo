/**
 * keywords: [viewport, 3d, three, optimized, performance]
 * 
 * Optimized 3D viewport with high-performance rendering
 */

export class SimpleViewport3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        
        // Camera setup
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(15, 10, 15);
        
        // Renderer with performance settings
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: false, // Disable for performance
            powerPreference: "high-performance"
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(5, 0, 5);
        
        // Lights
        this.scene.add(new THREE.AmbientLight(0x404040));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 5);
        this.scene.add(dirLight);
        
        // Grid
        this.gridSize = 10;
        this.grid = new THREE.GridHelper(this.gridSize, this.gridSize, 0x444444, 0x222222);
        this.scene.add(this.grid);
        
        // Agent - optimized mesh
        const agentGeometry = new THREE.SphereGeometry(0.3, 8, 8);
        const agentMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x4a9eff,
            emissive: 0x4a9eff,
            emissiveIntensity: 0.2
        });
        this.agent = new THREE.Mesh(agentGeometry, agentMaterial);
        this.agent.position.y = 0.3;
        this.scene.add(this.agent);
        
        // Trajectory - pre-allocated buffer for performance
        this.MAX_TRAIL_LENGTH = 2000;
        this.trajectoryGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.MAX_TRAIL_LENGTH * 3);
        this.trajectoryGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.trajectoryGeometry.setDrawRange(0, 0);
        
        const trajectoryMaterial = new THREE.LineBasicMaterial({ 
            color: 0x4a9eff,
            opacity: 0.8,
            transparent: true
        });
        this.trajectoryLine = new THREE.Line(this.trajectoryGeometry, trajectoryMaterial);
        this.scene.add(this.trajectoryLine);
        
        // Rewards - using regular meshes instead of instanced mesh to avoid buffer issues
        this.rewards = new THREE.Group();
        this.scene.add(this.rewards);
        this.rewardMeshes = new Map();
        
        // Data buffers
        this.trajectoryBuffer = null;
        this.rewardPositions = [];
        
        // Settings
        this.followAgent = false;
        this.showHeatmap = false;
        this.quality = 'medium';
        
        // Handle resize
        this.resizeObserver = new ResizeObserver(() => this.onResize());
        this.resizeObserver.observe(this.container);
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastFPSTime = performance.now();
        this.fps = 0;
        
        // Animation loop
        this.animate();
    }
    
    setData(data) {
        this.data = data;
        
        // Update grid size if provided
        if (data.metadata?.grid_size) {
            this.gridSize = Math.max(...data.metadata.grid_size);
            this.scene.remove(this.grid);
            this.grid = new THREE.GridHelper(this.gridSize, this.gridSize, 0x444444, 0x222222);
            this.scene.add(this.grid);
            
            this.camera.position.set(this.gridSize * 1.5, this.gridSize, this.gridSize * 1.5);
            this.controls.target.set(this.gridSize / 2, 0, this.gridSize / 2);
        }
        
        // Pre-process trajectory for fast access
        const length = data.trajectory.x.length;
        this.trajectoryBuffer = new Float32Array(length * 3);
        for (let i = 0; i < length; i++) {
            this.trajectoryBuffer[i * 3] = data.trajectory.x[i];
            this.trajectoryBuffer[i * 3 + 1] = 0.1;
            this.trajectoryBuffer[i * 3 + 2] = data.trajectory.y[i];
        }
        
        // Clear existing rewards
        this.rewards.clear();
        this.rewardMeshes.clear();
        
        // Pre-process rewards
        this.rewardPositions = [];
        const rewardMap = new Map();
        
        // First pass: find all reward locations
        data.trajectory.x.forEach((x, i) => {
            if (data.rewards[i] > 0) {
                const gridX = Math.floor(x);
                const gridY = Math.floor(data.trajectory.y[i]);
                const key = `${gridX},${gridY}`;
                
                // Store first collection time for each unique reward position
                if (!rewardMap.has(key)) {
                    rewardMap.set(key, {
                        x: gridX + 0.5,
                        z: gridY + 0.5,
                        firstCollectTime: i,
                        value: data.rewards[i]
                    });
                }
            }
        });
        
        // Create reward meshes
        const rewardGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.1, 8);
        const rewardMaterial = new THREE.MeshPhongMaterial({
            color: 0xffeb3b,
            emissive: 0xffeb3b,
            emissiveIntensity: 0.3
        });
        
        rewardMap.forEach((pos, key) => {
            const mesh = new THREE.Mesh(rewardGeometry, rewardMaterial.clone());
            mesh.position.set(pos.x, 0.05, pos.z);
            mesh.userData = { firstCollectTime: pos.firstCollectTime, key: key };
            mesh.visible = true; // Ensure visible on load
            this.rewards.add(mesh);
            this.rewardMeshes.set(key, mesh);
        });
        
        // Store reward positions for later use
        this.rewardPositions = Array.from(rewardMap.values());
    }
    
    update(time) {
        if (!this.data || !this.trajectoryBuffer) return;
        
        const maxIndex = Math.floor(this.data.trajectory.x.length - 1);
        const index = Math.min(Math.floor(time), maxIndex);
        
        // Update agent position with interpolation
        const t = time - Math.floor(time);
        const i1 = Math.floor(time);
        const i2 = Math.min(i1 + 1, maxIndex);
        
        if (i2 <= maxIndex) {
            this.agent.position.x = this.trajectoryBuffer[i1 * 3] * (1 - t) + this.trajectoryBuffer[i2 * 3] * t;
            this.agent.position.z = this.trajectoryBuffer[i1 * 3 + 2] * (1 - t) + this.trajectoryBuffer[i2 * 3 + 2] * t;
        } else {
            this.agent.position.x = this.trajectoryBuffer[index * 3];
            this.agent.position.z = this.trajectoryBuffer[index * 3 + 2];
        }
        
        // Update trajectory with LOD
        if (time > 0) {
            const trailStart = Math.max(0, index - this.MAX_TRAIL_LENGTH);
            const trailLength = index - trailStart;
            
            // Determine LOD based on quality setting
            const lodStep = this.quality === 'low' ? 4 : this.quality === 'medium' ? 2 : 1;
            
            const positions = this.trajectoryGeometry.attributes.position.array;
            let writeIndex = 0;
            
            for (let i = trailStart; i <= index && writeIndex < this.MAX_TRAIL_LENGTH; i += lodStep) {
                positions[writeIndex * 3] = this.trajectoryBuffer[i * 3];
                positions[writeIndex * 3 + 1] = this.trajectoryBuffer[i * 3 + 1];
                positions[writeIndex * 3 + 2] = this.trajectoryBuffer[i * 3 + 2];
                writeIndex++;
            }
            
            this.trajectoryGeometry.attributes.position.needsUpdate = true;
            this.trajectoryGeometry.setDrawRange(0, writeIndex);
        }
        
        // Update collected rewards more efficiently
        this.rewardMeshes.forEach((mesh, key) => {
            const collectionTime = mesh.userData.firstCollectTime;
            // Show reward if we haven't reached its collection time yet
            mesh.visible = time < collectionTime;
        });
        
        // Follow agent with damping
        if (this.followAgent && this.agent.position) {
            const targetX = this.agent.position.x;
            const targetZ = this.agent.position.z;
            this.controls.target.x += (targetX - this.controls.target.x) * 0.1;
            this.controls.target.z += (targetZ - this.controls.target.z) * 0.1;
            this.controls.target.y = 0; // Keep looking at ground level
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update FPS counter
        this.frameCount++;
        const currentTime = performance.now();
        if (currentTime - this.lastFPSTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFPSTime = currentTime;
            
            // Dispatch FPS event
            this.container.dispatchEvent(new CustomEvent('fps-update', { detail: this.fps }));
        }
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    resetCamera() {
        this.camera.position.set(15, 10, 15);
        this.controls.target.set(this.gridSize / 2, 0, this.gridSize / 2);
    }
    
    setQuality(level) {
        this.quality = level;
        switch(level) {
            case 'low':
                this.renderer.setPixelRatio(1);
                this.MAX_TRAIL_LENGTH = 500;
                break;
            case 'medium':
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
                this.MAX_TRAIL_LENGTH = 1000;
                break;
            case 'high':
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this.MAX_TRAIL_LENGTH = 2000;
                break;
        }
    }
    
    dispose() {
        this.resizeObserver.disconnect();
        this.renderer.dispose();
        this.controls.dispose();
        this.trajectoryGeometry.dispose();
        this.agent.geometry.dispose();
        this.trajectoryLine.material.dispose();
        this.agent.material.dispose();
        
        // Dispose reward meshes
        this.rewards.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    }
}