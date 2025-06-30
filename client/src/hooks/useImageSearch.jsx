import { useState, useCallback } from 'react';

export const useImageSearch = () => {
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState(null);

  const searchByText = useCallback(async (searchText) => {
    if (!searchText || searchText.trim() === '') {
      setResults([]);
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5005/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchText,
          type: 'text',
        }),
      }).catch(err => {
        console.warn("Could not connect to backend server for search:", err);
        return { ok: false, status: 0, statusText: "Could not connect to server" };
      });

      if (!response.ok) {
        if (response.status === 0) {
          console.warn("Backend server not available for search, returning empty results");
          setResults([]);
          setError("Backend server not available. Please ensure the server is running.");
          return;
        }
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error('Error searching images:', err);
      setError(err.message);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, []);

  const searchByImage = useCallback(async (imageFile) => {
    if (!imageFile) {
      setResults([]);
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await fetch('http://localhost:5005/search', {
        method: 'POST',
        body: formData,
      }).catch(err => {
        console.warn("Could not connect to backend server for image search:", err);
        return { ok: false, status: 0, statusText: "Could not connect to server" };
      });

      if (!response.ok) {
        if (response.status === 0) {
          console.warn("Backend server not available for image search, returning empty results");
          setResults([]);
          setError("Backend server not available. Please ensure the server is running.");
          return;
        }
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error('Error searching images:', err);
      setError(err.message);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  return {
    results,
    isSearching,
    error,
    searchByText,
    searchByImage,
    clearResults
  };
};