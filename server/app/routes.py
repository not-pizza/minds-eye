from flask import Blueprint, request, jsonify, Response, stream_with_context, current_app
import os
import boto3
from botocore.exceptions import NoCredentialsError
import uuid
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime, timedelta
from functools import lru_cache
from .modal_client import ModalClipClient
from .database import save_image, get_all_images, get_all_embeddings, get_image, delete_image, count_images
from .pca import (
    compute_and_save_pca, get_active_pca_model, project_embedding,
    should_recompute_pca, compute_pca
)

main = Blueprint('main', __name__)

# Helper function to get server URL


def get_server_url():
    # Try to get from environment or config
    server_url = os.getenv('SERVER_URL')
    if server_url:
        return server_url.rstrip('/')

    # Default for development
    return "http://localhost:5005"


# S3 configuration
print("Checking for AWS credentials")


def get_s3_client():
    """Get or create S3 client with proper credentials."""
    # Try loading .env file again directly in this function
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    print(f"Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)

    # Also try direct file reading as a fallback
    try:
        if os.path.exists(env_path):
            print("Reading .env file directly")
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Only set if not already set
                        if not os.getenv(key.strip()):
                            os.environ[key.strip()] = value.strip().strip(
                                "'").strip('"')
            print("Direct .env file reading completed")
        else:
            print(f".env file not found at {env_path}")
    except Exception as e:
        print(f"Error reading .env file: {e}")

    s3_params = {}

    # Get credentials from environment
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', '').strip()
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '').strip()
    # Default to us-east-2
    aws_region = os.getenv('AWS_REGION', 'us-east-2').strip()

    # Print environment variables for debugging
    print("Environment variables:")
    print(f"AWS_ACCESS_KEY_ID length: {len(aws_access_key)}")
    print(f"AWS_SECRET_ACCESS_KEY length: {len(aws_secret_key)}")
    print(f"AWS_REGION: {aws_region}")
    print(f"S3_BUCKET_NAME: {os.getenv('S3_BUCKET_NAME', '')}")

    if not aws_access_key or not aws_secret_key or len(aws_access_key) < 10 or len(aws_secret_key) < 10:
        print("AWS credentials missing or invalid - using mock client")
    else:
        s3_params.update({
            'aws_access_key_id': aws_access_key,
            'aws_secret_access_key': aws_secret_key,
            'region_name': aws_region
        })

    # Determine bucket name and check if it's set
    bucket_name = os.getenv('S3_BUCKET_NAME', '').strip()
    if not bucket_name:
        print("S3_BUCKET_NAME environment variable is not set - using mock client")

    try:
        # Create a new session with specific credentials to avoid profile conflicts
        session = boto3.session.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Create client from this session with specific configuration
        client = session.client(
            's3',
            # Force specific signature version
            config=boto3.session.Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            ),
            # Try with explicit endpoint URL
            endpoint_url=f'https://s3.{aws_region}.amazonaws.com'
        )

        # Test the connection
        if bucket_name:
            client.head_bucket(Bucket=bucket_name)
        print("S3 client created successfully with explicit credentials")
        return client
    except Exception as e:
        print(f"Failed to create S3 client: {e}")


BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'dev-bucket')

# Initialize the Modal CLIP client
modal_client = ModalClipClient()


@main.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

# Create a proxy endpoint for images to avoid CORS issues


@main.route('/images/<image_key>', methods=['GET'])
def serve_image(image_key):
    """Proxy endpoint for serving images from S3 with caching."""
    try:
        # Validate image_key to prevent path traversal
        if os.path.sep in image_key or '..' in image_key:
            return jsonify({'error': 'Invalid image key'}), 400

        try:
            # Get the file from S3
            response = get_s3_client().get_object(
                Bucket=BUCKET_NAME,
                Key=image_key
            )

            # Stream the file content back to the client

            # Get content type with proper fallback for images
            content_type = response.get(
                'ContentType', 'application/octet-stream')
            if content_type == 'application/octet-stream' and (
                    image_key.lower().endswith('.jpg') or image_key.lower().endswith('.jpeg')):
                content_type = 'image/jpeg'
            elif content_type == 'application/octet-stream' and image_key.lower().endswith('.png'):
                content_type = 'image/png'

            # Use streaming response to avoid loading entire file into memory
            def generate():
                chunk_size = 4096  # 4KB chunks
                stream = response['Body']
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            resp = Response(
                stream_with_context(generate()),
                mimetype=content_type
            )

            # Add caching headers - cache for 1 day
            resp.headers['Cache-Control'] = 'public, max-age=86400'
            # Simple ETag based on image key
            resp.headers['ETag'] = f"\"{image_key}\""

            return resp
        except Exception as e:
            print(f"Error retrieving from S3: {e}")
            return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        print(f"Error serving image: {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Generate a unique ID for the image
        image_id = str(uuid.uuid4())

        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{image_id}{file_extension}"

        # Save a copy of the file data for embedding generation
        file_data = file.read()
        file_size = len(file_data)

        # Generate vector embedding using Modal CLIP
        try:
            vector = modal_client.embed_image(file_data)
            print("Generated embedding")
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Use random vector as fallback
            vector = [np.random.uniform(-1, 1) for _ in range(512)]

        # Get active PCA model and project embedding
        try:
            pca_model = get_active_pca_model()

            if pca_model:
                # Project the vector to 3D using PCA
                projected_vector = project_embedding(vector, pca_model)
            else:
                # No PCA model yet - use simple projection
                # Just take the first 3 dimensions of the vector
                projected_vector = vector[:3]

                # If we have enough images, compute PCA
                image_count = count_images()
                if image_count >= 3:  # Need at least 3 for meaningful PCA
                    compute_and_save_pca()
        except Exception as e:
            print(f"Error with PCA: {e}")
            # Use first 3 dimensions as fallback
            projected_vector = vector[:3]

        # Reset the file pointer (needed for some operations)
        file.seek(0)

        # Upload to S3
        try:
            # Create BytesIO object from the file data we saved earlier
            from io import BytesIO
            file_obj = BytesIO(file_data)

            # Get fresh credentials from environment
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', '').strip()
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '').strip()
            aws_region = os.getenv('AWS_REGION', 'us-east-2').strip()

            # Print first and last few chars of keys for debugging (careful not to expose full keys)
            if len(aws_access_key) > 8:
                print(
                    f"Using Access Key: {aws_access_key[:4]}...{aws_access_key[-4:]}")
            if len(aws_secret_key) > 8:
                print(
                    f"Using Secret Key: {aws_secret_key[:4]}...{aws_secret_key[-4:]}")

            # Create a dedicated session for this upload with specific configuration
            session = boto3.session.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Configure S3 client with specific parameters to address signature issues
            upload_client = session.client(
                's3',
                # Force specific signature version
                config=boto3.session.Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}
                ),
                # Try with explicit endpoint URL
                endpoint_url=f'https://s3.{aws_region}.amazonaws.com'
            )

            # Upload using this specific client
            get_s3_client().put_object(
                Bucket=BUCKET_NAME,
                Key=unique_filename,
                Body=file_data,
                ContentType=file.content_type,
                CacheControl='max-age=86400'
            )
            print(
                f"Successfully uploaded {unique_filename} to S3 with explicit session")

            # Create the URL to access the image through our proxy
            server_url = get_server_url()
            presigned_url = f"{server_url}/images/{unique_filename}"
            s3_url = f"s3://{BUCKET_NAME}/{unique_filename}"
        except Exception as e:
            print(f"S3 upload failed: {e}")
            return jsonify({'error': 'Failed to upload image to S3'}), 500

        # Save to database
        try:
            # Save the image data and embedding to the database
            image_record = save_image(
                image_id=image_id,
                s3_key=unique_filename,
                s3_url=s3_url,  # Store the S3 path, not a public URL
                embedding=vector,
                content_type=file.content_type,
                metadata={
                    'original_filename': file.filename,
                    'size': file_size,
                    'uploaded_at': datetime.utcnow().isoformat()
                },
                projected_embedding=projected_vector
            )

            # Check if we should recompute PCA immediately after upload
            if should_recompute_pca():  # Use new interval-based logic
                print(f"Triggering immediate PCA recalculation after upload")
                pca_result = compute_and_save_pca()
                if pca_result:
                    print(f"PCA recalculated successfully. Model ID: {pca_result['id']}, embeddings: {pca_result['num_embeddings']}")
                else:
                    print("Failed to recalculate PCA")
        except Exception as e:
            print(f"Database error: {e}")
            # We still want to return some data even if DB fails

        # Return the image information
        return jsonify({
            'id': image_id,
            'url': presigned_url,  # Return pre-signed URL for immediate display
            'vector': projected_vector
        })

    except NoCredentialsError:
        return jsonify({'error': 'AWS credentials not available'}), 500
    except Exception as e:
        print(f"Unexpected error in upload: {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/get-vector', methods=['POST'])
def get_vector():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the file data
            file_data = file.read()

            try:
                # Generate vector embedding using Modal CLIP
                vector = modal_client.embed_image(file_data)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Use random vector as fallback
                vector = [np.random.uniform(-1, 1) for _ in range(512)]

            try:
                # Get active PCA model and project embedding
                pca_model = get_active_pca_model()

                if pca_model:
                    # Project the vector to 3D using PCA
                    projected_vector = project_embedding(vector, pca_model)
                else:
                    # No PCA model yet - use simple projection
                    projected_vector = vector[:3]
            except Exception as e:
                print(f"Error with PCA: {e}")
                # Use first 3 dimensions as fallback
                projected_vector = vector[:3]

            return jsonify({
                'vector': projected_vector,
                'full_vector': vector
            })
        except Exception as e:
            print(f"Unexpected error in get_vector: {e}")
            return jsonify({'error': str(e)}), 500


@main.route('/images', methods=['GET'])
def get_images():
    try:
        # Get all images from the database
        try:
            images = get_all_images()
        except Exception as e:
            print(f"Error getting images from database: {e}")
            images = []  # Use empty list as fallback

        # Get server URL for full URLs
        server_url = get_server_url()

        # Generate URLs for each image
        for image in images:
            # Use our proxy endpoint with full URL to avoid CORS issues
            image['url'] = f"{server_url}/images/{image.get('s3_key', 'unknown')}"

            # Use projected_vector if available, otherwise use the original vector
            if image.get('projected_vector'):
                image['position'] = image['projected_vector']
            elif image.get('vector'):
                # Fallback to the first 3 dimensions of the full vector
                image['position'] = image['vector'][:3]
            else:
                # Random position as a last resort
                image['position'] = [
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                ]

        return jsonify({'images': images})

    except Exception as e:
        print(f"Unexpected error in get_images: {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/images/<image_id>', methods=['GET'])
def get_image_by_id(image_id):
    try:
        try:
            image = get_image(image_id)
        except Exception as e:
            print(f"Error getting image from database: {e}")
            return jsonify({'error': 'Error retrieving image from database'}), 500

        if image:
            # Get server URL for full URLs
            server_url = get_server_url()
            image['url'] = f"{server_url}/images/{image.get('s3_key', 'unknown')}"

            return jsonify(image)
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        print(f"Unexpected error in get_image_by_id: {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/images/<image_id>', methods=['DELETE'])
def delete_image_by_id(image_id):
    try:
        # Get the image record to find the S3 key
        image = get_image(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        # Delete from S3
        get_s3_client().delete_object(
            Bucket=BUCKET_NAME,
            Key=image['s3_key']
        )

        # Delete from database
        success = delete_image(image_id)
        if success:
            return jsonify({'status': 'deleted'})
        else:
            return jsonify({'error': 'Failed to delete from database'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/search', methods=['POST'])
def search_images():
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            query_type = data.get('type', 'text')
            query = data.get('query')

            if not query:
                return jsonify({'error': 'No query provided'}), 400
        else:
            # Check if it's an image upload
            if 'image' in request.files:
                query_type = 'image'
                query = None  # Will be read from the file later
            else:
                # Default to text search from form
                query_type = request.form.get('type', 'text')
                query = request.form.get('query')

                if not query:
                    return jsonify({'error': 'No query provided'}), 400

        try:
            # Get all image embeddings from the database
            embeddings_data = get_all_embeddings()
        except Exception as e:
            print(f"Error getting embeddings from database: {e}")
            embeddings_data = []  # Use empty list as fallback

        if not embeddings_data:
            return jsonify({'results': []})

        # Extract embeddings and keep track of image info
        images = []
        embeddings = []

        for item in embeddings_data:
            if 'vector' in item and 's3_key' in item:
                images.append({
                    'id': item['id'],
                    's3_key': item.get('s3_key')
                })
                embeddings.append(item['vector'])

        if not embeddings:
            return jsonify({'results': []})

        try:
            # If query is an image file, process it
            if query_type == 'image' and 'image' in request.files:
                file = request.files['image']
                file_data = file.read()
                results = modal_client.search_similar_images(
                    file_data,
                    'image',
                    embeddings
                )
            else:
                # Text query
                results = modal_client.search_similar_images(
                    query,
                    'text',
                    embeddings
                )
        except Exception as e:
            print(f"Error searching with Modal: {e}")
            # Return random results as fallback
            import random
            results = [
                {'index': i, 'score': random.random()}
                for i in range(min(5, len(images)))
            ]
            random.shuffle(results)

        # Format results with image information and pre-signed URLs
        formatted_results = []
        for result in results:
            idx = result['index']
            if 0 <= idx < len(images):
                try:
                    # Use our proxy endpoint with full URL to avoid CORS issues
                    server_url = get_server_url()
                    presigned_url = f"{server_url}/images/{images[idx]['s3_key']}"
                except Exception as e:
                    print(f"Error generating presigned URL: {e}")
                    # Use S3 proxy as fallback
                    server_url = get_server_url()
                    presigned_url = f"{server_url}/images/{images[idx].get('s3_key', 'unknown')}"

                formatted_results.append({
                    'id': images[idx]['id'],
                    'url': presigned_url,
                    'score': result['score']
                })

        return jsonify({'results': formatted_results})

    except Exception as e:
        print(f"Unexpected error in search_images: {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/pca', methods=['GET'])
def get_pca_info():
    """Get information about the current PCA model."""
    try:
        pca_model = get_active_pca_model()
        if pca_model:
            return jsonify({
                'status': 'active',
                'id': pca_model['id'],
                'created_at': pca_model['created_at'],
                'num_embeddings': pca_model['num_embeddings'],
                'n_components': pca_model['n_components'],
                'explained_variance_ratio': pca_model.get('metadata', {}).get('explained_variance_ratio'),
                'singular_values': pca_model.get('metadata', {}).get('singular_values')
            })
        else:
            return jsonify({
                'status': 'none',
                'message': 'No active PCA model found'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/pca/recompute', methods=['POST'])
def recompute_pca():
    """Force recomputation of PCA model."""
    try:
        pca_model = compute_and_save_pca(force=True)
        if pca_model:
            return jsonify({
                'status': 'success',
                'message': 'PCA model recomputed successfully',
                'model_id': pca_model['id'],
                'num_embeddings': pca_model['num_embeddings']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to recompute PCA model'
            }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Legacy function for backward compatibility


def generate_vector_embedding(file):
    """
    Generate a vector embedding for an image using Modal CLIP.
    This function is maintained for backward compatibility.
    """
    try:
        # Read the file data
        if hasattr(file, 'read'):
            # If file is a file-like object
            file_position = file.tell()
            file_data = file.read()
            file.seek(file_position)  # Reset file position
        else:
            # If file is already bytes
            file_data = file

        # Generate vector embedding using Modal CLIP
        vector = modal_client.embed_image(file_data)

        # Get active PCA model and project embedding
        pca_model = get_active_pca_model()

        if pca_model:
            # Project the vector to 3D using PCA
            return project_embedding(vector, pca_model)
        else:
            # No PCA model yet - use simple projection
            return vector[:3]

    except Exception as e:
        print(f"Error generating vector: {e}")
        # Return a random vector as fallback
        return [np.random.random() for _ in range(3)]
