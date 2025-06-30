# Happy Ocean Times

An interactive 3D photo gallery application with privacy-focused design, using AWS S3 for secure storage, Amazon RDS for database, and Modal CLIP for advanced image embeddings.

## Project Structure

This is a monorepo containing:

- `client/`: React frontend application
- `server/`: Python Flask backend with AWS integrations and Modal CLIP

## Features

- Upload and view images in 3D space
- **Privacy-focused**: All images are stored privately in S3
- Secure access using pre-signed URLs that expire
- Intelligent image placement using CLIP embeddings
- Vector embeddings stored in Amazon RDS PostgreSQL
- Search images by text description
- Highlight search results in the 3D view

## Prerequisites

- Node.js (v14+)
- Python (v3.8+)
- AWS account with:
  - S3 bucket (private access)
  - RDS PostgreSQL database
- Modal account (for CLIP embeddings)

## Setup

### AWS Configuration

1. **Create a Private S3 Bucket**:
   - Go to AWS Management Console → S3
   - Click "Create bucket"
   - Enter a unique bucket name
   - Choose a region close to you
   - Under "Block Public Access settings" → Enable ALL blocking options
   - Create the bucket

2. Create an IAM user with S3 permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::your-private-bucket-name",
           "arn:aws:s3:::your-private-bucket-name/*"
         ]
       }
     ]
   }
   ```

3. Create an RDS PostgreSQL database using the provided script:
   ```bash
   cd server
   python scripts/create_rds.py
   ```

4. Copy `server/.env.example` to `server/.env` and fill in your AWS credentials

### Modal Configuration

1. Sign up for a Modal account at https://modal.com
2. Deploy the CLIP model using the provided `server/modal_clip.py` script:
   ```bash
   cd server
   modal deploy modal_clip.py
   ```
3. Add your Modal API key and endpoint URL to the `.env` file

### Server Setup

```bash
cd server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Client Setup

```bash
cd client
npm install
```

## Running the Application

From the root directory:

```bash
# Install root dependencies
npm install

# Run both client and server
npm start
```

Alternatively, you can run them separately:

```bash
# Start the client
npm run start:client

# Start the server
npm run start:server
```

## Privacy & Security Features

- **Private S3 Storage**: All images are stored in a private S3 bucket
- **Pre-signed URLs**: Temporary access URLs that expire after 1 hour
- **Automatic URL Refresh**: Client automatically refreshes URLs before they expire
- **No Public Access**: No direct public access to your images
- **Secure Flow**: Images are uploaded through your server, never directly to S3

## Usage

1. Open your browser to http://localhost:3000
2. Drag and drop images onto the canvas
3. Images will be uploaded to private S3, with embeddings stored in RDS
4. Images are displayed in 3D space based on their visual similarity
5. Search for images by entering text descriptions in the search box
6. Matching images will be highlighted with a yellow outline
7. Your images will persist between sessions and remain private

## Architecture

The application uses the following AWS services:
- **Amazon S3**: Privately stores the actual image files
- **Amazon RDS**: PostgreSQL database for storing metadata and image embeddings
- **Modal**: Hosts the Jina CLIP v2 model for generating image embeddings

The backend uses:
- **Flask**: Web server that handles API requests
- **SQLAlchemy**: ORM for database interactions
- **Boto3**: AWS SDK for Python
- **Pre-signed URLs**: For secure, time-limited access to private S3 objects

The frontend uses:
- **React**: UI framework
- **Three.js** (via react-three-fiber): 3D rendering
- **URL Refresh System**: Maintains access to images as pre-signed URLs expire