"""
Script to configure CORS settings for your S3 bucket.
You need to run this once to allow browsers to access images directly from S3.

To run:
1. Make sure you have valid AWS credentials configured
2. Update the BUCKET_NAME to match your S3 bucket name
3. Run: python3 configure_s3_cors.py
"""

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# Get the bucket name from .env
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Create S3 client
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Configure CORS
cors_configuration = {
    'CORSRules': [{
        'AllowedHeaders': ['*'],
        'AllowedMethods': ['GET', 'HEAD'],
        'AllowedOrigins': ['http://localhost:3000', 'https://*.your-domain.com'],
        'ExposeHeaders': ['ETag', 'Content-Length', 'Content-Type'],
        'MaxAgeSeconds': 3000
    }]
}

# Apply the CORS configuration to the bucket
try:
    s3.put_bucket_cors(Bucket=BUCKET_NAME, CORSConfiguration=cors_configuration)
    print(f"CORS configuration applied successfully to bucket: {BUCKET_NAME}")
    print("Now browsers will be able to load images directly from S3")
except Exception as e:
    print(f"Error configuring CORS: {e}")
    print("Possible causes:")
    print("1. Invalid AWS credentials")
    print("2. The bucket doesn't exist")
    print("3. You don't have permission to update the bucket configuration")