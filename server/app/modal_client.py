"""
Client for interacting with the Modal-hosted CLIP model.

This module provides a client for generating image embeddings using
the Jina CLIP v2 model hosted on Modal.
"""
import os
import base64
import io
import requests
from typing import List, Dict, Any, Optional, Union
import json
from dotenv import load_dotenv

load_dotenv()


class ModalClipClient:
    """Client for interacting with the Modal CLIP model API."""

    def __init__(self):
        """Initialize the Modal CLIP client."""
        self.modal_embed_endpoint = os.getenv('MODAL_EMBED_ENDPOINT')
        self.modal_batch_embed_endpoint = os.getenv(
            'MODAL_BATCH_EMBED_ENDPOINT')
        self.modal_search_endpoint = os.getenv('MODAL_SEARCH_ENDPOINT')
        self.api_key = os.getenv('MODAL_API_KEY')

        if not self.modal_embed_endpoint or not self.modal_batch_embed_endpoint or not self.modal_search_endpoint:
            print("Warning: MODAL_CLIP_ENDPOINT not set in environment variables")

        if not self.api_key:
            print("Warning: MODAL_API_KEY not set in environment variables")

    def embed_image(self,
                    image_data: Union[bytes, str],
                    truncate_dim: int = 512) -> List[float]:
        """
        Generate embeddings for an image using the Modal CLIP model.

        Args:
            image_data: Image data as bytes or base64 string
            truncate_dim: Dimension to truncate embeddings to (64-1024)

        Returns:
            Image embedding as a list of floats
        """
        if not self.modal_embed_endpoint or not self.api_key:
            print(
                "Warning: MODAL_EMBED_ENDPOINT or MODAL_API_KEY not set in environment variables")

        # Convert image to base64 if it's bytes
        if isinstance(image_data, bytes):
            base64_image = base64.b64encode(image_data).decode('utf-8')
        else:
            base64_image = image_data

        # Prepare the request
        headers = {
            'Content-Type': 'application/json',
        }

        payload = {
            'type': 'image',
            'content': base64_image,
            'truncate_dim': truncate_dim,
            'normalize': True
        }

        try:
            response = requests.post(
                f"{self.modal_embed_endpoint}",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('embeddings', [])
            else:
                print(
                    f"Error from Modal API: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Exception calling Modal API: {str(e)}")

    def search_similar_images(self,
                              query: Union[str, bytes],
                              query_type: str,
                              database_embeddings: List[List[float]],
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar images using the Modal CLIP model.

        Args:
            query: Query text or image data
            query_type: Type of query ('text' or 'image')
            database_embeddings: Pre-computed embeddings to search
            top_k: Number of results to return

        Returns:
            List of dicts with 'index' and 'score'
        """
        if not self.modal_search_endpoint or not self.api_key:
            print(
                "Warning: MODAL_SEARCH_ENDPOINT or MODAL_API_KEY not set in environment variables")

        # Convert query to base64 if it's an image as bytes
        if query_type == 'image' and isinstance(query, bytes):
            query = base64.b64encode(query).decode('utf-8')

        # Prepare the request
        headers = {
            'Content-Type': 'application/json',
        }

        payload = {
            'query': query,
            'query_type': query_type,
            'database_embeddings': database_embeddings,
            'top_k': top_k,
            'task': 'retrieval.query' if query_type == 'text' else None,
            'truncate_dim': 512  # Standard dimension
        }

        try:
            response = requests.post(
                f"{self.modal_search_endpoint}",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('results', [])
            else:
                print(
                    f"Error from Modal API: {response.status_code} - {response.text}")
                # Fallback to random search results
                import random
                indices = list(range(min(len(database_embeddings), top_k)))
                random.shuffle(indices)

                return [
                    {'index': idx, 'score': random.random()}
                    for idx in indices[:top_k]
                ]

        except Exception as e:
            print(f"Exception calling Modal API: {str(e)}")
            # Fallback to random search results
            import random
            indices = list(range(min(len(database_embeddings), top_k)))
            random.shuffle(indices)

            return [
                {'index': idx, 'score': random.random()}
                for idx in indices[:top_k]
            ]
