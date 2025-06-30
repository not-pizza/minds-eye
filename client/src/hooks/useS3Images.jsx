import { useEffect, useState } from "react";

export const useS3Images = () => {
  const [images, setImages] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchImages = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:5005/images")
        .catch(err => {
          console.warn("Could not connect to backend server:", err);
          return { ok: false, status: 0, statusText: "Could not connect to server" };
        });
      
      if (!response.ok) {
        if (response.status === 0) {
          console.warn("Backend server not available, using empty images list");
          setImages([]);
          setError("Backend server not available. Please ensure the server is running.");
          return;
        }
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Store images with full URLs
      const processedImages = (data.images || []).map(image => ({
        ...image,
        url: image.url && !image.url.startsWith('http') ? `http://localhost:5005${image.url}` : image.url
      }));
      
      setImages(processedImages);
      setError(null);
    } catch (err) {
      console.error("Error fetching images:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };



  useEffect(() => {
    fetchImages();
  }, []);

  const uploadImage = async (file) => {
    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("http://localhost:5005/upload", {
        method: "POST",
        body: formData,
      }).catch(err => {
        console.warn("Could not connect to backend server for upload:", err);
        return { ok: false, status: 0, statusText: "Could not connect to server" };
      });

      if (!response.ok) {
        if (response.status === 0) {
          console.warn("Backend server not available for upload, using placeholder data");
          // Create a placeholder image with random vector
          const placeholderId = Date.now().toString();
          // Use local browser URL directly since this is a fallback when server is unavailable
          const placeholderData = {
            id: placeholderId,
            url: URL.createObjectURL(file),
            vector: [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1],
          };
          
          // Add the placeholder image to the state
          setImages((prevImages) => [
            ...prevImages,
            {
              id: placeholderData.id,
              url: placeholderData.url,
              vector: placeholderData.vector,
              position: placeholderData.vector,
            },
          ]);
          
          return placeholderData;
        }
        
        const errorData = await response.json();
        throw new Error(errorData.error || "Upload failed");
      }

      const data = await response.json();
      
      // Format the URL with server base path if it's a relative URL
      const fullUrl = data.url && !data.url.startsWith('http') 
        ? `http://localhost:5005${data.url}` 
        : data.url;
      
      // Add the new image to the state
      setImages((prevImages) => [
        ...prevImages,
        {
          id: data.id,
          url: fullUrl,
          vector: data.vector,
          position: data.vector,
        },
      ]);
      

      return data;
    } catch (error) {
      console.error("Upload error:", error);
      throw error;
    }
  };

  return {
    images,
    isLoading,
    error,
    uploadImage,
    refreshImages: fetchImages
  };
};