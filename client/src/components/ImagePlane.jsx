import { useRef, useEffect, useState, useMemo } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export const ImagePlane = ({ position, path, highlight = false }) => {
  const planeRef = useRef();
  const outlineRef = useRef();
  const { camera } = useThree();
  const [aspectRatio, setAspectRatio] = useState(1);
  const [textureLoaded, setTextureLoaded] = useState(false);
  const [loadError, setLoadError] = useState(false);
  
  // Create texture manually instead of using useLoader to handle errors
  const imageTexture = useMemo(() => {
    const texture = new THREE.TextureLoader();
    
    // Check if path is a mock URL
    if (path && path.startsWith('mock-url-for-')) {
      // Create a canvas with a placeholder image
      const canvas = document.createElement('canvas');
      canvas.width = 200;
      canvas.height = 200;
      const ctx = canvas.getContext('2d');
      
      // Fill with a gradient
      const gradient = ctx.createLinearGradient(0, 0, 200, 200);
      gradient.addColorStop(0, '#8360c3');
      gradient.addColorStop(1, '#2ebf91');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 200, 200);
      
      // Add some text
      ctx.fillStyle = 'white';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Image Placeholder', 100, 100);
      
      const canvasTexture = new THREE.CanvasTexture(canvas);
      setTextureLoaded(true);
      return canvasTexture;
    }
    
    // Load the actual image
    return texture.load(
      path,
      (loadedTexture) => {
        setTextureLoaded(true);
        setLoadError(false);
      },
      undefined,
      (error) => {
        // Log detailed error information for debugging
        console.error(`Could not load image: ${path}`);
        console.error(`Error details:`, error);
        
        // Check if this is likely a CORS issue
        if (path && path.includes('localhost') && !path.startsWith('http')) {
          console.warn('Possible CORS issue: URL may need to be absolute with protocol');
        }
        
        setLoadError(true);
        
        // Create a canvas with an error message
        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 200;
        const ctx = canvas.getContext('2d');
        
        // Fill with a red background
        ctx.fillStyle = '#ffdddd';
        ctx.fillRect(0, 0, 200, 200);
        
        // Add error text
        ctx.fillStyle = '#ff0000';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Error loading image', 100, 100);
        
        // Show the URL that failed
        ctx.font = '10px Arial';
        const displayPath = path.length > 30 ? path.substring(0, 27) + '...' : path;
        ctx.fillText(displayPath, 100, 130);
        
        // Create a new texture from the error canvas
        const errorTexture = new THREE.CanvasTexture(canvas);
        
        // Replace the original texture
        if (planeRef.current) {
          planeRef.current.material.map = errorTexture;
          planeRef.current.material.needsUpdate = true;
        }
      }
    );
  }, [path]);

  useEffect(() => {
    if (planeRef.current && imageTexture && textureLoaded) {
      planeRef.current.material.map = imageTexture;
      imageTexture.needsUpdate = true;
      
      // Update aspect ratio to match the loaded image
      if (imageTexture.image) {
        const imgAspect = imageTexture.image.width / imageTexture.image.height;
        setAspectRatio(imgAspect);
        
        // Update the plane geometry with the correct aspect ratio
        if (planeRef.current) {
          planeRef.current.geometry = new THREE.PlaneGeometry(20 * imgAspect, 20);
        }
        if (outlineRef.current) {
          outlineRef.current.geometry = new THREE.PlaneGeometry(20 * imgAspect + 1, 20 + 1);
        }
      }
    }
  }, [imageTexture, textureLoaded]);

  // Create pulsing effect for highlighted images
  useFrame(({ clock }) => {
    if (planeRef.current) {
      // Make the plane always face the camera
      planeRef.current.lookAt(camera.position);
      
      if (outlineRef.current) {
        outlineRef.current.lookAt(camera.position);
        
        // If highlighted, create a pulsing effect
        if (highlight) {
          const pulse = Math.sin(clock.getElapsedTime() * 3) * 0.2 + 0.8;
          outlineRef.current.material.opacity = pulse;
          outlineRef.current.visible = true;
        } else {
          outlineRef.current.visible = false;
        }
      }
    }
  });

  return (
    <group position={position}>
      {/* Main image plane */}
      <mesh
        ref={planeRef}
        geometry={new THREE.PlaneGeometry(20 * aspectRatio, 20)}
        material={new THREE.MeshBasicMaterial({
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 1,
          map: imageTexture
        })}
      />
      
      {/* Outline plane (slightly larger, only visible when highlighted) */}
      <mesh
        ref={outlineRef}
        position={[0, 0, -0.1]} // Just behind the main plane
        geometry={new THREE.PlaneGeometry(20 * aspectRatio + 1, 20 + 1)}
        material={new THREE.MeshBasicMaterial({
          color: 0xffff00, // Yellow highlight
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.8,
          wireframe: false
        })}
        visible={highlight}
      />
    </group>
  );
};