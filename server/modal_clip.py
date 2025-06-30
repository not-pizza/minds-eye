"""
Deploy Jina CLIP v2 on Modal for multilingual multimodal embeddings.

Features:
- Multilingual support (89 languages)
- 512x512 image resolution
- Matryoshka embeddings (64-1024 dimensions)
- Text and image embedding endpoints
- Batch processing support
- Similarity search functionality
- Task-specific encoding (retrieval.query, retrieval.passage)
"""

import modal
from typing import List, Dict, Union, Optional, Literal
import numpy as np
from PIL import Image
import requests
import io
import base64

# Create Modal app
app = modal.App("jina-clip-v2")

# Define the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers==4.45.0",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "safetensors>=0.4.0",
        # Note: xformers removed due to version conflicts
        # It's optional - the model will work without it
    )
)

# Model configuration
MODEL_NAME = "jinaai/jina-clip-v2"

# Supported truncation dimensions for Matryoshka
VALID_DIMENSIONS = [64, 128, 256, 512, 768, 1024]

# Task types
TaskType = Literal["retrieval.query", "retrieval.passage",
                   "text-matching", "classification", "separation"]

# Create a volume for model caching
volume = modal.Volume.from_name("jina-clip-v2-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": volume},
    scaledown_window=300,
    max_containers=10,
)
class JinaCLIPv2Model:
    model = None
    device = None
    dtype = None

    @modal.enter()
    def load_model(self):
        """Load the Jina CLIP v2 model on container startup."""
        import torch
        from transformers import AutoModel

        # Detect device inside the container
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(
            f"Loading Jina CLIP v2 model on {self.device} with dtype {self.dtype}...")

        # Use cache directory for model storage
        cache_dir = "/cache/models"

        # Load model exactly as shown in the official documentation
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # Move to device and set dtype
        if self.device == "cuda":
            self.model = self.model.to(self.device, dtype=self.dtype)
        else:
            self.model = self.model.to(self.device)

        # Set to eval mode
        self.model.eval()

        print("Model loaded successfully!")

    def _validate_truncate_dim(self, truncate_dim: Optional[int]) -> Optional[int]:
        """Validate truncation dimension."""
        if truncate_dim is None:
            return None
        if truncate_dim not in VALID_DIMENSIONS:
            raise ValueError(
                f"Invalid truncate_dim: {truncate_dim}. "
                f"Must be one of {VALID_DIMENSIONS}"
            )
        return truncate_dim

    def _load_image(self, image_input: Union[str, bytes]) -> Union[Image.Image, str]:
        """Load image from URL, base64 string, or bytes."""
        # The model can handle URLs directly, so return them as-is
        if isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
            return image_input

        try:
            # Handle base64
            if isinstance(image_input, str):
                # Remove data URL prefix if present
                if ',' in image_input:
                    image_input = image_input.split(',')[1]
                image_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Handle raw bytes
            else:
                return Image.open(io.BytesIO(image_input)).convert('RGB')

        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    @modal.method()
    def embed_text(
        self,
        texts: Union[str, List[str]],
        task: Optional[TaskType] = None,
        truncate_dim: Optional[int] = None,
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text input(s).

        Note: The model returns normalized embeddings by default.
        The normalize parameter is kept for API compatibility but has no effect.
        """
        import torch
        import numpy as np

        truncate_dim = self._validate_truncate_dim(truncate_dim)

        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Apply task prefix if specified
        if task == "retrieval.query":
            texts = [
                "Represent the query for retrieving evidence documents: " + text for text in texts]
        elif task == "retrieval.passage":
            texts = ["Represent the passage for retrieval: " +
                     text for text in texts]
        elif task == "text-matching":
            texts = ["Represent the text for matching: " +
                     text for text in texts]
        elif task == "classification":
            texts = ["Represent the text for classification: " +
                     text for text in texts]
        elif task == "separation":
            texts = ["Represent the text for separation: " +
                     text for text in texts]

        # Generate embeddings using the model's encode_text method
        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts,
                truncate_dim=truncate_dim
            )

            # Handle both tensor and numpy array outputs
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Convert to list
            embeddings = embeddings.tolist()

        return embeddings[0] if single_input else embeddings

    @modal.method()
    def embed_image(
        self,
        images: Union[str, bytes, List[Union[str, bytes]]],
        truncate_dim: Optional[int] = None,
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for image input(s).

        Note: The model returns normalized embeddings by default.
        The normalize parameter is kept for API compatibility but has no effect.
        """
        import torch
        import numpy as np

        truncate_dim = self._validate_truncate_dim(truncate_dim)

        if not isinstance(images, list):
            images = [images]
            single_input = True
        else:
            single_input = False

        # Process images - the model can handle URLs, PIL images, etc.
        processed_images = [self._load_image(img) for img in images]

        # Generate embeddings using the model's encode_image method
        with torch.no_grad():
            embeddings = self.model.encode_image(
                processed_images,
                truncate_dim=truncate_dim
            )

            # Handle both tensor and numpy array outputs
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Convert to list
            embeddings = embeddings.tolist()

        return embeddings[0] if single_input else embeddings

    @modal.method()
    def embed_batch(
        self,
        items: List[Dict[str, any]],
        truncate_dim: Optional[int] = None,
        normalize: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for a batch of mixed text and image inputs."""
        results = []

        for item in items:
            if item['type'] == 'text':
                embedding = self._get_text_embedding(
                    item['content'],
                    task=item.get('task'),
                    truncate_dim=truncate_dim
                )
            elif item['type'] == 'image':
                embedding = self._get_image_embedding(
                    item['content'],
                    truncate_dim=truncate_dim
                )
            else:
                raise ValueError(f"Unknown type: {item['type']}")

            results.append(embedding)

        return results

    @modal.method()
    def compute_similarity(
        self,
        embeddings1: Union[List[float], List[List[float]]],
        embeddings2: Union[List[float], List[List[float]]]
    ) -> Union[float, List[float], List[List[float]]]:
        """Compute cosine similarity between embeddings."""
        return self._compute_similarity_internal(embeddings1, embeddings2)

    def _compute_similarity_internal(
        self,
        embeddings1: Union[List[float], List[List[float]]],
        embeddings2: Union[List[float], List[List[float]]]
    ) -> Union[float, List[float], List[List[float]]]:
        """Internal method to compute cosine similarity."""
        import numpy as np

        # Convert to numpy arrays
        emb1 = np.array(embeddings1)
        emb2 = np.array(embeddings2)

        # Ensure 2D arrays
        if emb1.ndim == 1:
            emb1 = emb1[np.newaxis, :]
        if emb2.ndim == 1:
            emb2 = emb2[np.newaxis, :]

        # Embeddings should already be normalized, just compute dot product
        similarities = np.dot(emb1, emb2.T)

        # Return appropriate format
        if similarities.shape == (1, 1):
            return float(similarities[0, 0])
        elif similarities.shape[0] == 1:
            return similarities[0].tolist()
        elif similarities.shape[1] == 1:
            return similarities[:, 0].tolist()
        else:
            return similarities.tolist()

    @modal.method()
    def search(
        self,
        query: Union[str, bytes],
        query_type: Literal["text", "image"],
        database_embeddings: List[List[float]],
        top_k: int = 10,
        task: Optional[TaskType] = "retrieval.query",
        truncate_dim: Optional[int] = None
    ) -> List[Dict[str, Union[int, float]]]:
        """Search for most similar items in a database."""
        import numpy as np

        # Get query embedding by calling the internal methods directly
        if query_type == 'text':
            # Call the method's internal logic directly
            query_embedding = self._get_text_embedding(
                query,
                task=task,
                truncate_dim=truncate_dim
            )
        else:
            query_embedding = self._get_image_embedding(
                query,
                truncate_dim=truncate_dim
            )

        # Truncate database embeddings if needed
        if truncate_dim:
            database_embeddings = [emb[:truncate_dim]
                                   for emb in database_embeddings]

        # Compute similarities using the internal method
        similarities = self._compute_similarity_internal(
            query_embedding, database_embeddings)

        # Get top-k indices
        if isinstance(similarities, (int, float)):
            similarities = [similarities]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(similarities[idx])
            })

        return results

    def _get_text_embedding(self, text: str, task: Optional[str] = None, truncate_dim: Optional[int] = None) -> List[float]:
        """Internal method to get text embedding."""
        import torch
        import numpy as np

        truncate_dim = self._validate_truncate_dim(truncate_dim)

        # Apply task prefix if specified
        if task == "retrieval.query":
            text = "Represent the query for retrieving evidence documents: " + text
        elif task == "retrieval.passage":
            text = "Represent the passage for retrieval: " + text
        elif task == "text-matching":
            text = "Represent the text for matching: " + text
        elif task == "classification":
            text = "Represent the text for classification: " + text
        elif task == "separation":
            text = "Represent the text for separation: " + text

        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_text(
                [text],
                truncate_dim=truncate_dim
            )

            # Handle both tensor and numpy array outputs
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            elif not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            # Return first element since we passed a single text
            return embedding[0].tolist()

    def _get_image_embedding(self, image: Union[str, bytes], truncate_dim: Optional[int] = None) -> List[float]:
        """Internal method to get image embedding."""
        import torch
        import numpy as np

        truncate_dim = self._validate_truncate_dim(truncate_dim)

        # Process image
        processed_image = self._load_image(image)

        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_image(
                [processed_image],
                truncate_dim=truncate_dim
            )

            # Handle both tensor and numpy array outputs
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            elif not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            # Return first element since we passed a single image
            return embedding[0].tolist()


# Create web endpoints as separate functions
@app.function(image=image, gpu="A10G", volumes={"/cache": volume})
@modal.fastapi_endpoint(method="POST")
def embed(request: Dict) -> Dict:
    """
    Web endpoint for embedding generation.

    Expected request format:
    {
        "type": "text" or "image",
        "content": str, List[str], or base64,
        "task": Optional[str] (for text),
        "truncate_dim": Optional[int] (64-1024),
        "normalize": Optional[bool] (default True)
    }

    Returns:
    {
        "embeddings": List[float] or List[List[float]],
        "dimension": int,
        "model": str
    }
    """
    model = JinaCLIPv2Model()

    content = request["content"]
    content_type = request.get("type", "text")
    truncate_dim = request.get("truncate_dim")
    normalize = request.get("normalize", True)

    if content_type == "text":
        task = request.get("task")
        embeddings = model.embed_text.remote(
            content,
            task=task,
            truncate_dim=truncate_dim,
            normalize=normalize
        )
    else:
        embeddings = model.embed_image.remote(
            content,
            truncate_dim=truncate_dim,
            normalize=normalize
        )

    # Determine actual dimension
    if isinstance(embeddings, list) and isinstance(embeddings[0], list):
        dimension = len(embeddings[0])
    else:
        dimension = len(embeddings)

    return {
        "embeddings": embeddings,
        "dimension": dimension,
        "model": MODEL_NAME
    }


@app.function(image=image, gpu="A10G", volumes={"/cache": volume})
@modal.fastapi_endpoint(method="POST")
def search(request: Dict) -> Dict:
    """
    Web endpoint for similarity search.

    Expected request format:
    {
        "query": str or base64,
        "query_type": "text" or "image",
        "database_embeddings": List[List[float]],
        "top_k": Optional[int] (default 10),
        "task": Optional[str] (default "retrieval.query" for text),
        "truncate_dim": Optional[int]
    }

    Returns:
    {
        "results": List[{"index": int, "score": float}]
    }
    """
    model = JinaCLIPv2Model()

    results = model.search.remote(
        query=request["query"],
        query_type=request["query_type"],
        database_embeddings=request["database_embeddings"],
        top_k=request.get("top_k", 10),
        task=request.get(
            "task", "retrieval.query" if request["query_type"] == "text" else None),
        truncate_dim=request.get("truncate_dim")
    )

    return {"results": results}


@app.function(image=image, gpu="A10G", volumes={"/cache": volume})
@modal.fastapi_endpoint(method="POST")
def batch_embed(request: Dict) -> Dict:
    """
    Web endpoint for batch embedding generation.

    Expected request format:
    {
        "items": [
            {
                "type": "text" or "image",
                "content": str or base64,
                "task": Optional[str] (for text)
            },
            ...
        ],
        "truncate_dim": Optional[int],
        "normalize": Optional[bool] (default True)
    }

    Returns:
    {
        "embeddings": List[List[float]],
        "dimension": int,
        "count": int
    }
    """
    model = JinaCLIPv2Model()

    embeddings = model.embed_batch.remote(
        items=request["items"],
        truncate_dim=request.get("truncate_dim"),
        normalize=request.get("normalize", True)
    )

    return {
        "embeddings": embeddings,
        "dimension": len(embeddings[0]) if embeddings else 0,
        "count": len(embeddings)
    }


# CLI for testing
@app.local_entrypoint()
def main():
    """Test the deployment with sample data."""
    try:
        # Create an instance for testing
        model = JinaCLIPv2Model()

        # Test text embeddings in multiple languages
        print("Testing multilingual text embeddings...")
        texts = [
            "A beautiful sunset over the beach",  # English
            "Un beau coucher de soleil sur la plage",  # French
            "海滩上美丽的日落",  # Chinese
            "Прекрасный закат над пляжем",  # Russian
        ]

        # Note: normalize=True by default
        text_embeddings = model.embed_text.remote(
            texts, task="retrieval.passage", truncate_dim=512)
        print(
            f"Generated {len(text_embeddings)} text embeddings of dimension {len(text_embeddings[0])}")

        # Test image embeddings
        print("\nTesting image embeddings...")
        image_urls = [
            "https://i.ibb.co/nQNGqL0/beach1.jpg",
            "https://i.ibb.co/r5w8hG8/beach2.jpg"
        ]

        image_embeddings = model.embed_image.remote(
            image_urls, truncate_dim=512)
        print(
            f"Generated {len(image_embeddings)} image embeddings of dimension {len(image_embeddings[0])}")

        # Test cross-modal similarity
        print("\nTesting cross-modal similarity...")
        query = "beautiful sunset beach"
        query_embedding = model.embed_text.remote(
            query, task="retrieval.query", truncate_dim=512)

        # Compute similarities
        text_similarities = model.compute_similarity.remote(
            query_embedding, text_embeddings)
        image_similarities = model.compute_similarity.remote(
            query_embedding, image_embeddings)

        print(f"Text-to-text similarities: {text_similarities}")
        print(f"Text-to-image similarities: {image_similarities}")

        # Test search
        print("\nTesting search functionality...")
        all_embeddings = text_embeddings + image_embeddings
        search_results = model.search.remote(
            query=query,
            query_type="text",
            database_embeddings=all_embeddings,
            top_k=3,
            task="retrieval.query",
            truncate_dim=512
        )

        print("Top 3 search results:")
        for result in search_results:
            item_type = "text" if result["index"] < len(texts) else "image"
            print(
                f"  - {item_type} index {result['index']}: score {result['score']:.4f}")

        print("\nDeployment test completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
