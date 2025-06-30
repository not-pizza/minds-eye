"""
PCA module for dimensionality reduction of image embeddings.

This module provides functionality to:
1. Calculate PCA from high-dimensional CLIP embeddings
2. Project embeddings to 3D space for visualization
3. Store and retrieve PCA components and other statistics
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from datetime import datetime

from .database import get_db_session, get_all_embeddings, update_image_embedding

# Create a custom model to store PCA data
from sqlalchemy import Column, String, Float, DateTime, Text, Integer
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from .database import Base, engine

logger = logging.getLogger(__name__)

class PCAModel(Base):
    """Model for storing PCA components and statistics."""
    __tablename__ = 'pca_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, server_default=func.now())
    num_embeddings = Column(Integer, nullable=False)
    n_components = Column(Integer, nullable=False)
    
    # Store the PCA components as JSON arrays
    components = Column(Text, nullable=False)  # JSON string of PCA components
    explained_variance = Column(Text, nullable=False)  # JSON string of explained variance
    mean = Column(Text, nullable=False)  # JSON string of mean vector
    
    # Metadata
    is_active = Column(Integer, default=1)  # 1 = active, 0 = inactive
    model_metadata = Column(Text)  # Additional metadata as JSON string

    def to_dict(self) -> Dict[str, Any]:
        """Convert the PCA model to a dictionary."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'num_embeddings': self.num_embeddings,
            'n_components': self.n_components,
            'components': json.loads(self.components),
            'explained_variance': json.loads(self.explained_variance),
            'mean': json.loads(self.mean),
            'is_active': bool(self.is_active),
            'metadata': json.loads(self.model_metadata) if self.model_metadata else None
        }

# Create tables if they don't exist
Base.metadata.create_all(engine)

def compute_pca(embeddings: List[List[float]], n_components: int = 3) -> Tuple[PCA, np.ndarray]:
    """
    Compute PCA on a set of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        n_components: Number of PCA components to keep
        
    Returns:
        Tuple of (PCA model, projected embeddings)
    """
    if not embeddings or len(embeddings) < 2:
        raise ValueError("Need at least 2 embeddings to compute PCA")
    
    # Convert to numpy array
    X = np.array(embeddings)
    
    # Create and fit PCA model
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(X)
    
    return pca, projected

def save_pca_model(pca_model: PCA, num_embeddings: int, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Save PCA model to the database.
    
    Args:
        pca_model: Fitted scikit-learn PCA model
        num_embeddings: Number of embeddings used to fit the model
        metadata: Additional metadata to store
        
    Returns:
        The saved PCA model record as a dictionary
    """
    session = get_db_session()
    try:
        # Deactivate previous models
        session.query(PCAModel).update({'is_active': 0})
        
        # Convert numpy arrays to lists for JSON serialization
        components = pca_model.components_.tolist()
        explained_variance = pca_model.explained_variance_.tolist()
        mean = pca_model.mean_.tolist()
        
        # Create new model record
        pca_record = PCAModel(
            num_embeddings=num_embeddings,
            n_components=pca_model.n_components_,
            components=json.dumps(components),
            explained_variance=json.dumps(explained_variance),
            mean=json.dumps(mean),
            is_active=1,
            model_metadata=json.dumps(metadata) if metadata else None
        )
        
        session.add(pca_record)
        session.commit()
        
        return pca_record.to_dict()
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving PCA model: {e}")
        raise
    finally:
        session.close()

def get_active_pca_model() -> Optional[Dict[str, Any]]:
    """
    Get the currently active PCA model.
    
    Returns:
        The active PCA model as a dictionary, or None if no model exists
    """
    session = get_db_session()
    try:
        pca_record = session.query(PCAModel).filter(PCAModel.is_active == 1).first()
        if pca_record:
            return pca_record.to_dict()
        return None
    except Exception as e:
        logger.error(f"Error retrieving PCA model: {e}")
        return None
    finally:
        session.close()

def recreate_sklearn_pca(pca_dict: Dict[str, Any]) -> PCA:
    """
    Recreate a scikit-learn PCA object from the stored dictionary.
    
    Args:
        pca_dict: Dictionary representation of a PCA model
        
    Returns:
        A scikit-learn PCA object
    """
    pca = PCA(n_components=pca_dict['n_components'])
    
    # Set attributes manually
    pca.components_ = np.array(pca_dict['components'])
    pca.explained_variance_ = np.array(pca_dict['explained_variance'])
    pca.mean_ = np.array(pca_dict['mean'])
    
    # Calculate additional attributes
    pca.explained_variance_ratio_ = pca.explained_variance_ / np.sum(pca.explained_variance_)
    
    return pca

def project_embedding(embedding: List[float], pca_model: Optional[Dict[str, Any]] = None) -> List[float]:
    """
    Project a single embedding using the PCA model.
    
    Args:
        embedding: The embedding vector to project
        pca_model: Optional PCA model dictionary (if not provided, will fetch from DB)
        
    Returns:
        The projected embedding (3D coordinates)
    """
    if not pca_model:
        pca_model = get_active_pca_model()
        
    if not pca_model:
        # No PCA model available, return the first 3 dimensions or random values
        if len(embedding) >= 3:
            return embedding[:3]
        else:
            return list(np.random.random(3))
    
    # Recreate scikit-learn PCA object
    pca = recreate_sklearn_pca(pca_model)
    
    # Project the embedding
    projected = pca.transform([embedding])[0]
    
    return projected.tolist()

def compute_and_save_pca(force: bool = False) -> Optional[Dict[str, Any]]:
    """
    Compute PCA from all embeddings and save the model.
    
    Args:
        force: If True, compute PCA even if there are few embeddings
        
    Returns:
        The saved PCA model as a dictionary, or None if computation failed
    """
    try:
        # Get all embeddings from the database
        embeddings_data = get_all_embeddings()
        
        if not embeddings_data or len(embeddings_data) < 3:
            if not force:
                logger.info("Not enough embeddings to compute meaningful PCA (need at least 3)")
                return None
        
        # Extract just the embedding vectors
        embedding_vectors = []
        for item in embeddings_data:
            if 'vector' in item:
                embedding_vectors.append(item['vector'])
        
        # Compute PCA
        pca_model, projected_embeddings = compute_pca(embedding_vectors)
        
        # Save the PCA model
        saved_model = save_pca_model(
            pca_model, 
            len(embedding_vectors),
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'explained_variance_ratio': pca_model.explained_variance_ratio_.tolist(),
                'singular_values': pca_model.singular_values_.tolist() if hasattr(pca_model, 'singular_values_') else None
            }
        )
        
        # Update all image embeddings with their projected values
        for i, item in enumerate(embeddings_data):
            if 'id' in item and i < len(projected_embeddings):
                # Update the image record with the projected embedding
                update_image_embedding(
                    item['id'],
                    projected_embeddings[i].tolist(),
                    field='projected_embedding'  # Store in a separate field
                )
        
        return saved_model
    except Exception as e:
        logger.error(f"Error computing and saving PCA: {e}")
        return None

def should_recompute_pca() -> bool:
    """
    Determine if PCA should be recomputed based on specific intervals.
    
    PCA is recalculated at these image counts:
    1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1500, 2000, 3000, 5000, 10000
    
    Returns:
        True if PCA should be recomputed, False otherwise
    """
    # Define the intervals at which PCA should be recalculated
    PCA_INTERVALS = [
        1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        1500, 2000, 3000, 5000, 10000
    ]
    
    try:
        # Get the active PCA model
        active_model = get_active_pca_model()
        
        # Get current number of embeddings
        embeddings_data = get_all_embeddings()
        current_count = len(embeddings_data)
        
        if not active_model:
            # No model exists, should compute one if we have any embeddings
            return current_count > 0
        
        # Get number of embeddings used for the active model
        model_count = active_model['num_embeddings']
        
        # Check if current count matches any of our target intervals
        # and is greater than the model count (meaning we've added images since last PCA)
        if current_count > model_count and current_count in PCA_INTERVALS:
            logger.info(f"PCA recalculation triggered: current_count={current_count}, model_count={model_count}")
            return True
        
        # Also check if we've passed an interval without recalculating
        # Find the largest interval <= current_count
        target_interval = None
        for interval in reversed(PCA_INTERVALS):
            if current_count >= interval:
                target_interval = interval
                break
        
        # If we found a target interval and our model is from before that interval, recalculate
        if target_interval and model_count < target_interval:
            logger.info(f"PCA recalculation triggered: reached interval {target_interval}, model_count={model_count}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error checking if PCA should be recomputed: {e}")
        return False