"""
Database module for storing image embeddings in Amazon RDS.
"""
import os
import json
import logging
import sqlalchemy
from sqlalchemy import create_engine, Column, String, Float, LargeBinary, Text, DateTime, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.sql import func
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Get database connection details from environment variables
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')

# Create the database connection string
if all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    logger.warning("Database environment variables not set properly. Using SQLite for development.")
    DATABASE_URL = "sqlite:///happy_ocean_times.db"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for declarative models
Base = declarative_base()

class Image(Base):
    """Model for storing image data and embeddings."""
    __tablename__ = 'images'
    
    id = Column(String(36), primary_key=True)  # UUID
    s3_key = Column(String(255), nullable=False)
    s3_url = Column(String(500), nullable=False)
    content_type = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Store the embedding as a JSON array
    # For PostgreSQL, we could use the ARRAY type, but this is more portable
    embedding = Column(Text, nullable=False)  # JSON string of embedding values
    
    # Store the projected embedding (via PCA) as a JSON array
    projected_embedding = Column(Text)  # JSON string of projected values
    
    # Additional metadata as JSON - renamed from 'metadata' to avoid conflict
    image_metadata = Column(Text)  # JSON string for additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert the image record to a dictionary."""
        return {
            'id': self.id,
            's3_key': self.s3_key,
            'url': self.s3_url,
            'content_type': self.content_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'vector': json.loads(self.embedding) if self.embedding else None,
            'projected_vector': json.loads(self.projected_embedding) if self.projected_embedding else None,
            'metadata': json.loads(self.image_metadata) if self.image_metadata else None
        }


# Create all tables
def init_db():
    """Initialize the database, creating tables if they don't exist."""
    Base.metadata.create_all(engine)
    logger.info("Database tables created")


# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """Get a database session."""
    session = SessionLocal()
    try:
        return session
    except Exception as e:
        session.rollback()
        raise e


def save_image(
    image_id: str,
    s3_key: str,
    s3_url: str,
    embedding: List[float],
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projected_embedding: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Save image data and embedding to the database.
    
    Args:
        image_id: UUID for the image
        s3_key: S3 object key
        s3_url: Public URL for the S3 object
        embedding: Vector embedding from CLIP
        content_type: MIME type of the image
        metadata: Additional metadata
        projected_embedding: Optional 3D projection of the embedding
        
    Returns:
        The saved image record as a dictionary
    """
    session = get_db_session()
    try:
        # Convert embedding to JSON string
        embedding_json = json.dumps(embedding)
        
        # Convert projected embedding to JSON string if present
        projected_json = json.dumps(projected_embedding) if projected_embedding else None
        
        # Convert metadata to JSON string if present
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Create the image record
        image = Image(
            id=image_id,
            s3_key=s3_key,
            s3_url=s3_url,
            content_type=content_type,
            embedding=embedding_json,
            projected_embedding=projected_json,
            image_metadata=metadata_json  # Changed from metadata to image_metadata
        )
        
        session.add(image)
        session.commit()
        
        return image.to_dict()
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving image to database: {e}")
        raise
    finally:
        session.close()


def get_image(image_id: str) -> Optional[Dict[str, Any]]:
    """
    Get an image record by ID.
    
    Args:
        image_id: UUID of the image
        
    Returns:
        The image record as a dictionary, or None if not found
    """
    session = get_db_session()
    try:
        image = session.query(Image).filter(Image.id == image_id).first()
        if image:
            return image.to_dict()
        return None
    except Exception as e:
        logger.error(f"Error retrieving image from database: {e}")
        return None
    finally:
        session.close()


def get_all_images() -> List[Dict[str, Any]]:
    """
    Get all image records.
    
    Returns:
        List of image records as dictionaries
    """
    session = get_db_session()
    try:
        images = session.query(Image).all()
        return [image.to_dict() for image in images]
    except Exception as e:
        logger.error(f"Error retrieving images from database: {e}")
        return []
    finally:
        session.close()


def get_all_embeddings() -> List[Dict[str, Any]]:
    """
    Get all image embeddings.
    
    Returns:
        List of dictionaries with image ID, URL, and embedding vector
    """
    session = get_db_session()
    try:
        images = session.query(Image).all()
        return [
            {
                'id': image.id,
                'url': image.s3_url,
                's3_key': image.s3_key,
                'vector': json.loads(image.embedding) if image.embedding else None,
                'projected_vector': json.loads(image.projected_embedding) if image.projected_embedding else None
            }
            for image in images
        ]
    except Exception as e:
        logger.error(f"Error retrieving embeddings from database: {e}")
        return []
    finally:
        session.close()


def delete_image(image_id: str) -> bool:
    """
    Delete an image record by ID.
    
    Args:
        image_id: UUID of the image
        
    Returns:
        True if deletion was successful, False otherwise
    """
    session = get_db_session()
    try:
        image = session.query(Image).filter(Image.id == image_id).first()
        if image:
            session.delete(image)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting image from database: {e}")
        return False
    finally:
        session.close()


def update_image_embedding(image_id: str, embedding: List[float], field: str = 'embedding') -> bool:
    """
    Update the embedding for an existing image.
    
    Args:
        image_id: UUID of the image
        embedding: New vector embedding
        field: Which embedding field to update ('embedding' or 'projected_embedding')
        
    Returns:
        True if update was successful, False otherwise
    """
    session = get_db_session()
    try:
        image = session.query(Image).filter(Image.id == image_id).first()
        if image:
            # Update the specified embedding field
            embedding_json = json.dumps(embedding)
            if field == 'embedding':
                image.embedding = embedding_json
            elif field == 'projected_embedding':
                image.projected_embedding = embedding_json
            else:
                raise ValueError(f"Invalid embedding field: {field}")
                
            image.updated_at = datetime.utcnow()
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating image embedding: {e}")
        return False
    finally:
        session.close()


def count_images() -> int:
    """
    Count the number of images in the database.
    
    Returns:
        The number of images
    """
    session = get_db_session()
    try:
        return session.query(Image).count()
    except Exception as e:
        logger.error(f"Error counting images: {e}")
        return 0
    finally:
        session.close()


# Initialize the database when the module is imported
init_db()