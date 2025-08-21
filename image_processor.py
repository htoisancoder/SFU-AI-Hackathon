import cv2
import numpy as np
import sqlite3
import math
import time
import threading
import requests
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Configuration
GRID_SIZE = 14
BLEED_RADIUS = 6
DB_PATH = "image_features.db"

# Color thresholds
NEAR_BLACK_THRESHOLD = 15
NEAR_WHITE_THRESHOLD = 240
LOW_SATURATION_THRESHOLD = 25

# Lock for database access
db_lock = threading.Lock()

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

def circular_mean_hue(hues):
    """Calculate circular mean for hue values"""
    rads = np.deg2rad(hues * 2)  # Convert to 0-360 range
    mean_cos = np.mean(np.cos(rads))
    mean_sin = np.mean(np.sin(rads))
    mean_rad = math.atan2(mean_sin, mean_cos)
    mean_deg = math.degrees(mean_rad) % 360 / 2
    return mean_deg

def load_image_with_unicode(path):
    """Load image supporting Unicode paths and URLs"""
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

def process_image(image_path):
    """Process image into 16x16 HSV chunks with bleeding"""
    img = load_image_with_unicode(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate chunk dimensions
    chunk_width = width / GRID_SIZE
    chunk_height = height / GRID_SIZE
    
    # Create empty array for chunk features
    features = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    
    # Process each chunk
    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            # Calculate chunk boundaries with bleeding
            x_start = max(0, int(grid_x * chunk_width) - BLEED_RADIUS)
            x_end = min(width, int((grid_x + 1) * chunk_width) + BLEED_RADIUS)
            y_start = max(0, int(grid_y * chunk_height) - BLEED_RADIUS)
            y_end = min(height, int((grid_y + 1) * chunk_height) + BLEED_RADIUS)
            
            # Extract chunk region
            chunk = img_hsv[y_start:y_end, x_start:x_end]
            
            # Split HSV channels
            h_vals = chunk[..., 0].flatten()
            s_vals = chunk[..., 1].flatten()
            v_vals = chunk[..., 2].flatten()
            
            # Skip processing if no pixels
            if len(h_vals) == 0:
                continue
            
            # Calculate means
            mean_v = np.mean(v_vals)
            mean_s = np.mean(s_vals)
            
            # Handle near-black chunks
            if mean_v < NEAR_BLACK_THRESHOLD:
                features[grid_y, grid_x] = [0, 0, mean_v]
                continue
            
            # Handle near-white, low-saturation chunks
            if mean_v > NEAR_WHITE_THRESHOLD and mean_s < LOW_SATURATION_THRESHOLD:
                features[grid_y, grid_x] = [0, 0, mean_v]
                continue
            
            # Calculate circular mean for hue
            mean_h = circular_mean_hue(h_vals)
            
            # Linear means for saturation and value
            mean_s = np.mean(s_vals)
            mean_v = np.mean(v_vals)
            
            features[grid_y, grid_x] = [mean_h, mean_s, mean_v]
    
    return {
        'path': str(image_path),
        'dims': (height, width),
        'features': features
    }

def init_db(db_path=DB_PATH):
    """Initialize SQLite database"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            height INTEGER,
            width INTEGER
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            image_id INTEGER,
            grid_row INTEGER,
            grid_col INTEGER,
            hue REAL,
            saturation REAL,
            value REAL,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )""")
    conn.commit()
    return conn

def store_features(conn, data):
    """Store features in database with thread safety"""
    with db_lock:
        cur = conn.cursor()
        path = data['path']
        
        # Check if image exists
        cur.execute("SELECT id FROM images WHERE path=?", (path,))
        existing = cur.fetchone()
        if existing:
            return existing[0], {'meta': 0.0, 'chunks': 0.0}
        
        # Insert image metadata
        height, width = data['dims']
        cur.execute("""
            INSERT INTO images (path, height, width)
            VALUES (?, ?, ?)
        """, (path, height, width))
        img_id = cur.lastrowid
        
        # Insert chunks
        chunks = []
        features = data['features']
        grid_h, grid_w = GRID_SIZE, GRID_SIZE
        
        for i in range(grid_h):
            for j in range(grid_w):
                h, s, v = features[i, j]
                # Convert numpy float32 to native Python float
                h = float(h)
                s = float(s)
                v = float(v)
                chunks.append((img_id, i, j, h, s, v))
        
        cur.executemany("""
            INSERT INTO chunks (image_id, grid_row, grid_col, hue, saturation, value)
            VALUES (?, ?, ?, ?, ?, ?)
        """, chunks)
        conn.commit()
        
        return img_id, {
            'meta': 0.0,
            'chunks': 0.0
        }

def main():
    # Set UTF-8 encoding for console output
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path> [image_path2 ...]")
        print("       python image_processor.py <directory_path>")
        print("Example: python image_processor.py my_photo.jpg")
        print("         python image_processor.py folder/")
        print("         python image_processor.py img1.jpg img2.png")
        sys.exit(1)
    
    # Start overall timer
    start_total = time.perf_counter()
    
    # Collect all image paths
    image_paths = []
    for path in sys.argv[1:]:
        path_obj = Path(path)
        if path_obj.is_dir():
            # Add all image files in directory
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
                image_paths.extend(path_obj.glob(ext))
        elif path_obj.exists():
            image_paths.append(path_obj)
        else:
            print(f"⚠️ Warning: Path not found - {path}")
    
    if not image_paths:
        print("❌ Error: No valid images found")
        sys.exit(1)
    
    print(f"\n{' STARTING BATCH PROCESSING ':=^50}")
    print(f"Found {len(image_paths)} images to process")
    print(f"Chunk grid: {GRID_SIZE}x{GRID_SIZE} chunks")
    print(f"Color bleeding: {BLEED_RADIUS} pixel radius")
    
    # Initialize database
    conn = init_db()
    
    # Process all images using thread pool
    results = []
    processed_count = 0
    total_images = len(image_paths)
    
    with ThreadPoolExecutor() as executor:
        # Submit all image processing tasks
        future_to_path = {executor.submit(process_image, path): path for path in image_paths}
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            processed_count += 1
            try:
                features = future.result()
                if not features:
                    print(f"❌ Failed to process: {path}")
                    continue
                    
                print(f"\n{' STORING IMAGE ' + str(processed_count) + '/' + str(total_images) + ' ':=^50}")
                print(f"Image: {path}")
                
                # Store features
                img_id, storage_timing = store_features(conn, features)
                
                results.append({
                    'path': path,
                    'id': img_id,
                    'features': features,
                    'storage_timing': storage_timing
                })
                
            except Exception as e:
                print(f"❌ Error processing {path}: {str(e)}")
    
    # Close database connection
    conn.close()
    
    # Calculate total time
    total_time = time.perf_counter() - start_total
    
    # Show individual results
    print("\n" + " IMAGE DETAILS " .center(50, '-'))
    for res in results:
        height, width = res['features']['dims']
        print(f"{Path(res['path']).name} ({width}x{height}) -> ID: {res['id']}")

    # Show final summary
    print("\n" + " BATCH PROCESSING SUMMARY " .center(50, '='))
    print(f"Total images processed: {len(results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/max(1, len(results)):.2f} seconds")
    
    # Pause at end
    print("\n" + "="*50)
    input("Processing complete. Press Enter to exit...")

@app.route('/process', methods=['POST'])
def process_image_endpoint():
    """HTTP endpoint to process an image"""
    # Check if the post request has the file part
    if 'file' not in request.files and 'url' not in request.form:
        return jsonify({'error': 'No file or URL provided'}), 400
    
    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Process the image
        result = process_image(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
    
    # Handle URL
    elif 'url' in request.form:
        url = request.form['url']
        if not is_url(url):
            return jsonify({'error': 'Invalid URL'}), 400
        
        # Process the image from URL
        result = process_image(url)
    
    if not result:
        return jsonify({'error': 'Failed to process image'}), 500
    
    # Store in database
    conn = init_db()
    try:
        img_id, timing = store_features(conn, result)
        conn.close()
        
        return jsonify({
            'success': True,
            'image_id': img_id,
            'path': result['path'],
            'dimensions': f"{result['dims'][1]}x{result['dims'][0]}"
        })
    except Exception as e:
        conn.close()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Start the Flask app
    print("Starting image processor service on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)