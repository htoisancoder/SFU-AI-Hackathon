import sqlite3
import cv2
import numpy as np
import math
import time
import concurrent.futures
from functools import partial
import requests
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Configuration - MUST match processor settings
GRID_SIZE = 14
BLEED_RADIUS = 6
DB_PATH = "image_features.db"

# Color thresholds
NEAR_BLACK_THRESHOLD = 15
NEAR_WHITE_THRESHOLD = 240
LOW_SATURATION_THRESHOLD = 25

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def is_url(path):
    """Check if the given path is a URL"""
    try:
        result = urlparse(path)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def download_image_from_url(url):
    """Download image from HTTP/HTTPS URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create a temporary file to save the image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        
        # Write content to temp file
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"❌ Error downloading image from {url}: {str(e)}")
        return None

def load_image_with_unicode(path):
    """Load image supporting Unicode paths and URLs on all platforms"""
    # Handle HTTP URLs
    if is_url(path):
        local_path = download_image_from_url(path)
        if not local_path:
            return None
        path = local_path
    
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"❌ Error: File not found: {path}")
        return None
    
    # Read as bytes to support Unicode paths
    with open(path_obj, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Decode image
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Error: Could not decode image: {path}")
    
    # Clean up temporary file if it was downloaded from URL
    if is_url(path) and os.path.exists(path):
        os.unlink(path)
    
    return img

def circular_mean_hue(hues):
    """Calculate circular mean for hue values"""
    if len(hues) == 0:
        return 0.0
    rads = np.deg2rad(hues * 2)  # Convert to 0-360 range
    mean_cos = np.mean(np.cos(rads))
    mean_sin = np.mean(np.sin(rads))
    mean_rad = math.atan2(mean_sin, mean_cos)
    mean_deg = math.degrees(mean_rad) % 360 / 2
    return mean_deg

def circular_distance(h1, h2):
    """Calculate circular distance between two hue values (0-180 range)"""
    diff = np.abs(h1 - h2)
    return np.minimum(diff, 180 - diff)

def chunk_similarity_vectorized(q_features, db_features):
    """Vectorized chunk similarity calculation"""
    # Extract components
    q_h, q_s, q_v = q_features[..., 0], q_features[..., 1], q_features[..., 2]
    db_h, db_s, db_v = db_features[..., 0], db_features[..., 1], db_features[..., 2]
    
    # Create mask for valid chunks (non-zero and non-NaN)
    valid_mask = (
        (np.sum(q_features, axis=2) != 0) & 
        (np.sum(db_features, axis=2) != 0) &
        (~np.isnan(q_features).any(axis=2)) & 
        (~np.isnan(db_features).any(axis=2))
    )
    
    # Create masks for special cases (near-black, near-white, low saturation)
    q_near_black = q_v < NEAR_BLACK_THRESHOLD
    q_near_white = q_v > NEAR_WHITE_THRESHOLD
    q_low_sat = q_s < LOW_SATURATION_THRESHOLD
    
    db_near_black = db_v < NEAR_BLACK_THRESHOLD
    db_near_white = db_v > NEAR_WHITE_THRESHOLD
    db_low_sat = db_s < LOW_SATURATION_THRESHOLD
    
    # Hue similarity (circular)
    h_dist = circular_distance(q_h, db_h)
    h_sim = 1 - h_dist / 180.0
    
    # Saturation similarity
    s_sim = 1 - np.abs(q_s - db_s) / 255.0
    
    # Value similarity
    v_sim = 1 - np.abs(q_v - db_v) / 255.0
    
    # Adjust weights based on special cases
    # For near-black or near-white/low-saturation regions, reduce hue importance
    q_special = q_near_black | (q_near_white & q_low_sat)
    db_special = db_near_black | (db_near_white & db_low_sat)
    
    # Create weight arrays
    h_weight = np.where(q_special | db_special, 0.2, 0.6)
    s_weight = np.where(q_special | db_special, 0.2, 0.2)
    v_weight = np.where(q_special | db_special, 0.6, 0.2)
    
    # Weighted average
    weighted_sim = (h_weight * h_sim) + (s_weight * s_sim) + (v_weight * v_sim)
    
    # Apply mask and calculate average
    valid_sims = weighted_sim[valid_mask]
    if valid_sims.size == 0:
        return 0.0
    return np.mean(valid_sims)

def load_database(db_path):
    """Load all image features from database"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Verify tables exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
    if not cur.fetchone():
        raise ValueError("Database missing 'images' table - run processor first")
    
    # Get all images
    cur.execute("SELECT id, path, height, width FROM images")
    images = {}
    for row in cur.fetchall():
        img_id, path, height, width = row
        images[img_id] = {
            'path': path,
            'dims': (height, width),
            'features': np.zeros((GRID_SIZE, GRID_SIZE, 3))
        }
    
    # Get all chunks
    cur.execute("SELECT image_id, grid_row, grid_col, hue, saturation, value FROM chunks")
    for row in cur.fetchall():
        img_id, row_idx, col_idx, h, s, v = row
        images[img_id]['features'][row_idx, col_idx] = [h, s, v]
    
    conn.close()
    return images

# Precompute diagonal order
def generate_diagonal_order(size):
    """Generate diagonal scanning order with 90-degree rotation offsets"""
    order = []
    # Calculate diagonal starting points
    for diag in range(2 * size - 1):
        if diag < size:
            start_row = size - 1 - diag
            start_col = size - 1
        else:
            start_row = 0
            start_col = 2 * size - 2 - diag
        
        # Process diagonal from bottom-right to top-left
        row, col = start_row, start_col
        while row < size and col >= 0:
            order.append((row, col))
            row += 1
            col -= 1
    
    return order

DIAGONAL_ORDER = generate_diagonal_order(GRID_SIZE)

def process_query_image(image_path):
    """Process query image using diagonal scanning order with bleeding compensation"""
    img = load_image_with_unicode(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create feature array
    features = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    chunk_w = width / GRID_SIZE
    chunk_h = height / GRID_SIZE
    
    # Process chunks in diagonal order
    for grid_y, grid_x in DIAGONAL_ORDER:
        # Calculate chunk boundaries with bleeding
        x_start = max(0, int(grid_x * chunk_w) - BLEED_RADIUS)
        x_end = min(width, int((grid_x + 1) * chunk_w) + BLEED_RADIUS)
        y_start = max(0, int(grid_y * chunk_h) - BLEED_RADIUS)
        y_end = min(height, int((grid_y + 1) * chunk_h) + BLEED_RADIUS)
        
        # Extract chunk
        chunk = img_hsv[y_start:y_end, x_start:x_end]
        if chunk.size == 0:
            continue
            
        # Split channels and convert to float32
        h_vals = chunk[..., 0].astype(np.float32).flatten()
        s_vals = chunk[..., 1].astype(np.float32).flatten()
        v_vals = chunk[..., 2].astype(np.float32).flatten()
        
        # Skip if no pixels
        if len(h_vals) == 0:
            continue
            
        # Handle near-black chunks
        mean_v = np.mean(v_vals)
        mean_s = np.mean(s_vals)
        
        if mean_v < NEAR_BLACK_THRESHOLD:
            features[grid_y, grid_x] = [0, 0, mean_v]
            continue
        
        # Handle near-white, low-saturation chunks
        if mean_v > NEAR_WHITE_THRESHOLD and mean_s < LOW_SATURATION_THRESHOLD:
            features[grid_y, grid_x] = [0, 0, mean_v]
            continue
        
        # Circular mean for hue
        h_rad = np.deg2rad(h_vals * 2)  # Convert to 0-360 range
        mean_cos = np.mean(np.cos(h_rad))
        mean_sin = np.mean(np.sin(h_rad))
        mean_h = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360 / 2
        
        # Linear means for saturation and value
        mean_s = np.mean(s_vals)
        mean_v = np.mean(v_vals)
        
        features[grid_y, grid_x] = [mean_h, mean_s, mean_v]
    
    return {
        'path': image_path,
        'dims': (height, width),
        'features': features
    }

def compute_similarity(img_id, db_data, q_features_list):
    """Compute max similarity for a database image across all rotations"""
    db_features = db_data['features']
    max_sim = 0.0
    for q_features in q_features_list:
        similarity = chunk_similarity_vectorized(q_features, db_features)
        if similarity > max_sim:
            max_sim = similarity
    return (img_id, max_sim)

def search_similar(q_features_list, db_images, top_k=10, workers=6):
    """Find top_k most similar images using parallel processing"""
    results = []
    
    # Create partial function for parallel execution
    compute_partial = partial(compute_similarity, q_features_list=q_features_list)
    
    # Use thread pool for parallel comparisons
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all comparison tasks
        future_to_id = {
            executor.submit(compute_partial, img_id, db_data): img_id
            for img_id, db_data in db_images.items()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_id):
            try:
                results.append(future.result())
            except Exception as e:
                img_id = future_to_id[future]
                print(f"❌ Error processing image {img_id}: {str(e)}")
                results.append((img_id, 0.0))
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

@app.route('/search', methods=['POST'])
def search_image():
    """HTTP endpoint to search for similar images"""
    # Check if the post request has the file part
    if 'file' not in request.files and 'url' not in request.form:
        return jsonify({'error': 'No file or URL provided'}), 400
    
    # Get top_k parameter
    top_k = int(request.form.get('top_k', 10))
    
    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Process the query image
        query_data = process_query_image(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
    
    # Handle URL
    elif 'url' in request.form:
        url = request.form['url']
        if not is_url(url):
            return jsonify({'error': 'Invalid URL'}), 400
        
        # Process the query image from URL
        query_data = process_query_image(url)
    
    if not query_data:
        return jsonify({'error': 'Failed to process query image'}), 500
    
    # Verify query image has non-zero values
    if np.sum(query_data['features']) == 0:
        return jsonify({'error': 'Query image processing resulted in all zeros'}), 500
    
    # Load database
    try:
        db_images = load_database(DB_PATH)
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    # Generate rotated feature grids for query
    q_features_0 = query_data['features']
    # 90° clockwise: transpose axes 0 and 1, then reverse columns
    q_features_90 = q_features_0.transpose(1, 0, 2)[:, ::-1, :]
    # 180°: reverse both axes
    q_features_180 = q_features_0[::-1, ::-1, :]
    # 270° clockwise: transpose axes 0 and 1, then reverse rows
    q_features_270 = q_features_0.transpose(1, 0, 2)[::-1, :, :]
    q_features_list = [q_features_0, q_features_90, q_features_180, q_features_270]
    
    # Perform search
    search_results = search_similar(q_features_list, db_images, top_k, workers=4)
    
    # Format results
    results = []
    for img_id, similarity in search_results:
        img_data = db_images[img_id]
        results.append({
            'image_id': img_id,
            'similarity': float(similarity),
            'path': img_data['path'],
            'dimensions': f"{img_data['dims'][1]}x{img_data['dims'][0]}"
        })
    
    return jsonify({
        'success': True,
        'results': results,
        'query_dimensions': f"{query_data['dims'][1]}x{query_data['dims'][0]}"
    })

if __name__ == "__main__":
    print("Starting image search service on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)