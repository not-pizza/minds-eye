import { Canvas } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "@react-three/drei";
import { ImagePlane } from "./components/ImagePlane";
import { ImageProcessor } from "./utils/imageProcessor";
import { useS3Images } from "./hooks/useS3Images";
import { useImageSearch } from "./hooks/useImageSearch";

const App = () => {
  const [displayedImages, setDisplayedImages] = useState([]);
  const [searchText, setSearchText] = useState("");
  const [highlightedImages, setHighlightedImages] = useState(new Set());
  const { images, isLoading, uploadImage, refreshImages } = useS3Images();
  const { 
    results: searchResults, 
    isSearching, 
    searchByText, 
    clearResults 
  } = useImageSearch();
  const searchInputRef = useRef(null);

  // Process images when they load from S3
  useEffect(() => {
    if (!isLoading && images.length > 0) {
      // Create 3D positions for images that don't have vectors
      const processedImages = images.map(image => {
        // If the image has a vector from the server, use it
        // Otherwise, we'll generate a random position
        const position = image.vector || [
          (Math.random() - 0.5) * 2,
          (Math.random() - 0.5) * 2,
          (Math.random() - 0.5) * 2,
        ];
        
        return {
          url: image.url,
          position: position
        };
      });
      
      setDisplayedImages(processedImages);
    }
  }, [images, isLoading]);

  // Update highlighted images when search results change
  useEffect(() => {
    if (searchResults && searchResults.length > 0) {
      const urlSet = new Set(searchResults.map(result => result.url));
      setHighlightedImages(urlSet);
    } else {
      setHighlightedImages(new Set());
    }
  }, [searchResults]);

  const handleImageProcessed = useCallback((imageData, index) => {
    // This function is called when a new image is processed
    // and will add the image to the displayed images
    if (imageData && imageData.url && imageData.vector) {
      setDisplayedImages(prev => [
        ...prev,
        {
          url: imageData.url,
          position: imageData.vector
        }
      ]);
    }
  }, []);

  const imageProcessor = useMemo(() => {
    const processor = ImageProcessor.getInstance()
      .setUploadFunction(uploadImage)
      .setOnImageProcessed(handleImageProcessed);
    return processor;
  }, [uploadImage, handleImageProcessed]);

  const onDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    const files = event.dataTransfer.files;
    for (let file of files) {
      if (file && file.type.startsWith("image/")) {
        imageProcessor.enqueue(file);
      }
    }
  };

  const onDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleReset = async () => {
    // In a real application, you would implement an API endpoint
    // to delete all images from S3
    setDisplayedImages([]);
    clearResults();
    setSearchText("");
    await refreshImages();
  };

  const handleSearch = async (event) => {
    event.preventDefault();
    if (searchText.trim()) {
      await searchByText(searchText);
    } else {
      clearResults();
    }
  };

  return (
    <div
      id="canvas-container"
      onDrop={onDrop}
      onDragOver={onDragOver}
      style={{ width: "100vw", height: "100vh" }}
    >
      <div className="controls" style={{ position: "absolute", top: 10, left: 10, zIndex: 100 }}>
        <button onClick={handleReset} style={{ marginRight: "10px" }}>Reset</button>
        
        <form onSubmit={handleSearch} style={{ display: "inline-flex" }}>
          <input
            ref={searchInputRef}
            type="text"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search images..."
            style={{ marginRight: "5px", padding: "5px" }}
          />
          <button type="submit" disabled={isSearching}>
            {isSearching ? "Searching..." : "Search"}
          </button>
          {searchResults.length > 0 && (
            <button 
              type="button" 
              onClick={() => {
                clearResults();
                setSearchText("");
              }} 
              style={{ marginLeft: "5px" }}
            >
              Clear
            </button>
          )}
        </form>
      </div>

      <Canvas
        camera={{ position: [0, 0, 50] }}
        gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
        linear
      >
        <OrbitControls minZoom={0} maxZoom={Infinity} />
        {displayedImages.map((image, index) => {
          if (!image || !image.position) {
            return null;
          }
          
          // Determine if this image is highlighted from search
          const isHighlighted = highlightedImages.has(image.url);
          
          return (
            <ImagePlane
              key={index}
              position={[
                image.position[0] * 750,
                image.position[1] * 750,
                image.position[2] * 750,
              ]}
              path={image.url}
              highlight={isHighlighted}
            />
          );
        })}
      </Canvas>
    </div>
  );
};

export default App;