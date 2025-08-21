import os
import sqlite3
import tempfile
import threading
from datetime import datetime
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from PIL import Image
from werkzeug.utils import secure_filename

# Import your modules
from image_processor import process_image, init_db as init_features_db
from image_searcher import process_query_image, load_database, search_similar
from tagging import Predictor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 4096 * 4096  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database configuration
DATABASE = 'images.db'
FEATURES_DB = 'image_features.db'

# Initialize the tag predictor
tag_predictor = Predictor()
USERS_DB = 'users.db'

def init_users_db():
    """Initialize the users database"""
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin BOOLEAN DEFAULT FALSE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def get_users_db_connection():
    """Get a users database connection"""
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    return conn

    # Add login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login', next=request.url))
        
        conn = get_users_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE id = ?', (session['user_id'],)
        ).fetchone()
        conn.close()
        
        if not user or not user['is_admin']:
            flash('Admin privileges required.', 'error')
            return redirect(url_for('index'))
            
        return f(*args, **kwargs)
    return decorated_function

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

def init_db():
    """Initialize the main database"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags TEXT,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            user_id INTEGER,  -- ✅ Add this line
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            count INTEGER DEFAULT 1
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dark_mode INTEGER DEFAULT 0
        )
    ''')
    # Initialize settings if not exists
    c.execute('SELECT COUNT(*) FROM settings')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO settings (dark_mode) VALUES (0)')
    conn.commit()
    conn.close()

def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_dark_mode_setting():
    """Get dark mode setting from database"""
    conn = get_db_connection()
    setting = conn.execute('SELECT dark_mode FROM settings WHERE id = 1').fetchone()
    conn.close()
    return setting['dark_mode'] if setting else 0

def update_dark_mode_setting(value):
    """Update dark mode setting in database"""
    conn = get_db_connection()
    conn.execute('UPDATE settings SET dark_mode = ? WHERE id = 1', (value,))
    conn.commit()
    conn.close()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_async(image_path, image_id, filename):
    """Process image in background thread for features and tags"""
    try:
        # Process for reverse image search features
        features_result = process_image(image_path)
        if features_result:
            # Store in features database
            features_conn = sqlite3.connect(FEATURES_DB)
            features_cursor = features_conn.cursor()
            
            # Check if image exists in features DB
            features_cursor.execute("SELECT id FROM images WHERE path=?", (filename,))
            existing = features_cursor.fetchone()
            
            if not existing:
                # Insert image metadata
                height, width = features_result['dims']
                features_cursor.execute(
                    "INSERT INTO images (path, height, width) VALUES (?, ?, ?)",
                    (filename, height, width)
                )
                img_features_id = features_cursor.lastrowid
                
                # Insert chunks
                chunks = []
                features = features_result['features']
                for i in range(14):  # GRID_SIZE
                    for j in range(14):  # GRID_SIZE
                        h, s, v = features[i, j]
                        chunks.append((img_features_id, i, j, float(h), float(s), float(v)))
                
                features_cursor.executemany(
                    "INSERT INTO chunks (image_id, grid_row, grid_col, hue, saturation, value) VALUES (?, ?, ?, ?, ?, ?)",
                    chunks
                )
                features_conn.commit()
            
            features_conn.close()
        
        # Process for tags
        img = Image.open(image_path).convert("RGBA")
        tag_result = tag_predictor.predict(img)
        
        # Update main database with tags
        conn = get_db_connection()
        conn.execute(
            "UPDATE images SET tags = ? WHERE id = ?",
            (tag_result['tags_string'], image_id)
        )
        conn.commit()
        conn.close()
        
        print(f"Successfully processed image {image_id}")
        
    except Exception as e:
        print(f"Error processing image {image_id}: {str(e)}")

@app.route('/')
def index():
    """Home page - show latest images"""
    conn = get_db_connection()
    images = conn.execute(
        'SELECT * FROM images ORDER BY upload_date DESC LIMIT 24'
    ).fetchall()
    conn.close()
    
    dark_mode = get_dark_mode_setting()
    return render_template('index.html', images=images, dark_mode=dark_mode)

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    if value is None:
        return "Unknown date"
    
    if isinstance(value, str):
        try:
            # Try to parse the string as datetime
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If parsing fails, return the original string
            return value
    elif isinstance(value, datetime):
        # It's already a datetime object
        pass
    else:
        # For other types, convert to string
        return str(value)
    
    return value.strftime(format)

@app.route('/toggle-dark-mode', methods=['POST'])
def toggle_dark_mode():
    """Toggle dark mode setting"""
    current_mode = get_dark_mode_setting()
    new_mode = 1 if current_mode == 0 else 0
    update_dark_mode_setting(new_mode)
    return jsonify({'success': True, 'dark_mode': new_mode})

@app.route('/check_image')
def check_image():
    """Check if an image exists in the database"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'exists': False})
    
    conn = get_db_connection()
    image = conn.execute(
        'SELECT * FROM images WHERE filename = ?', (filename,)
    ).fetchone()
    conn.close()
    
    return jsonify({'exists': image is not None})

@app.route('/save_uploaded_image', methods=['POST'])
def save_uploaded_image():
    """Save an uploaded image to the database permanently"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Check if already in database
    conn = get_db_connection()
    existing = conn.execute(
        'SELECT * FROM images WHERE filename = ?', (filename,)
    ).fetchone()
    
    if existing:
        conn.close()
        return jsonify({'message': 'Image already in database', 'image_id': existing['id']})
    
    # Add to database
    file_size = os.path.getsize(filepath)
    with Image.open(filepath) as img:
        width, height = img.size
    
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO images (filename, width, height, file_size) VALUES (?, ?, ?, ?)',
        (filename, width, height, file_size)
    )
    image_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Process image in background
    thread = threading.Thread(
        target=process_image_async, 
        args=(filepath, image_id, filename)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Image added to database', 'image_id': image_id})

@app.route('/upload', methods=['GET', 'POST'])
@login_required  # Add this decorator
def upload_file():
    """Upload new image - now requires login"""
    dark_mode = get_dark_mode_setting()
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            
            # Get image dimensions
            with Image.open(filepath) as img:
                width, height = img.size
            
            # Store in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
        'INSERT INTO images (filename, width, height, file_size, user_id) VALUES (?, ?, ?, ?, ?)',
        (filename, width, height, file_size, session['user_id'])  # Add user_id
    )
            image_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Process image in background (features and tags)
            thread = threading.Thread(
                target=process_image_async, 
                args=(filepath, image_id, filename)
            )
            thread.daemon = True
            thread.start()
            
            flash('Image successfully uploaded and is being processed', 'success')
            return redirect(url_for('image_detail', image_id=image_id))
    
    return render_template('upload.html', dark_mode=dark_mode)

@app.route('/image/<int:image_id>')
def image_detail(image_id):
    """Display detailed view of a single image"""
    conn = get_db_connection()
    image = conn.execute(
        'SELECT * FROM images WHERE id = ?', (image_id,)
    ).fetchone()
    conn.close()
    
    if not image:
        # Check if this is the next image request (N+1)
        next_id = image_id + 1
        conn = get_db_connection()
        next_image = conn.execute(
            'SELECT * FROM images WHERE id = ?', (next_id,)
        ).fetchone()
        conn.close()
        
        if next_image:
            # Redirect to the next available image
            return redirect(url_for('image_detail', image_id=next_id))
        else:
            flash('Image not found.', 'error')
            return redirect(url_for('index'))
    
    dark_mode = get_dark_mode_setting()
    is_owner = False
    if 'user_id' in session and image['user_id'] == session['user_id']:
        is_owner = True
    
    # Pass is_owner to the template
    return render_template('image_detail.html', image=image, dark_mode=dark_mode, is_owner=is_owner)

@app.route('/search')
def search():
    """Text-based search page"""
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = 24
    
    conn = get_db_connection()
    
    if query:
        # Simple tag-based search
        images = conn.execute(
            'SELECT * FROM images WHERE tags LIKE ? ORDER BY upload_date DESC LIMIT ? OFFSET ?',
            (f'%{query}%', per_page, (page-1)*per_page)
        ).fetchall()
        
        total = conn.execute(
            'SELECT COUNT(*) FROM images WHERE tags LIKE ?',
            (f'%{query}%',)
        ).fetchone()[0]
    else:
        # Show all images if no query
        images = conn.execute(
            'SELECT * FROM images ORDER BY upload_date DESC LIMIT ? OFFSET ?',
            (per_page, (page-1)*per_page)
        ).fetchall()
        
        total = conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    
    conn.close()
    
    dark_mode = get_dark_mode_setting()
    return render_template('search.html', 
                         images=images, 
                         query=query, 
                         page=page, 
                         per_page=per_page,
                         total=total,
                         total_pages=(total + per_page - 1) // per_page,
                         dark_mode=dark_mode)

@app.route('/reverse_search', methods=['GET', 'POST'])
def reverse_search():
    """Reverse image search page"""
    dark_mode = get_dark_mode_setting()
    
    if request.method == 'GET':
        image_id = request.args.get('image_id')
        if image_id:
            try:
                # Get the image to use for reverse search
                conn = get_db_connection()
                image = conn.execute(
                    'SELECT * FROM images WHERE id = ?', (image_id,)
                ).fetchone()
                conn.close()
                
                if image:
                    # Use the actual file from uploads folder
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                    if os.path.exists(image_path):
                        # Perform the reverse search using the actual file
                        return perform_reverse_search(image_path, dark_mode)
                    else:
                        flash('Image file not found', 'error')
            except Exception as e:
                flash(f'Error loading image: {str(e)}', 'error')
        
        return render_template('reverse_search.html', dark_mode=dark_mode)
    
    elif request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file directly to uploads folder
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Check if file already exists in database to avoid duplicates
                conn = get_db_connection()
                existing_image = conn.execute(
                    'SELECT * FROM images WHERE filename = ?', (filename,)
                ).fetchone()
                
                if existing_image:
                    # Use existing image ID
                    image_id = existing_image['id']
                else:
                    # Get file size and dimensions
                    file_size = os.path.getsize(filepath)
                    with Image.open(filepath) as img:
                        width, height = img.size
                    
                    # Add to database
                    cursor = conn.cursor()
                    cursor.execute(
                        'INSERT INTO images (filename, width, height, file_size) VALUES (?, ?, ?, ?)',
                        (filename, width, height, file_size)
                    )
                    image_id = cursor.lastrowid
                    conn.commit()
                    
                    # Process image in background for features and tags
                    thread = threading.Thread(
                        target=process_image_async, 
                        args=(filepath, image_id, filename)
                    )
                    thread.daemon = True
                    thread.start()
                
                conn.close()
                
                # Get top_k parameter
                top_k = int(request.form.get('top_k', 10))
                
                # Perform the search using the saved file
                result = perform_reverse_search(filepath, dark_mode, top_k)
                
                return result
                
            except Exception as e:
                return jsonify({'error': f'Search error: {str(e)}'}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400

def perform_reverse_search(image_path, dark_mode, top_k=10):
    """Perform reverse image search for a given image path"""
    try:
        # Process the query image
        query_data = process_query_image(image_path)
        
        if not query_data:
            flash('Failed to process query image', 'error')
            return redirect(url_for('reverse_search'))
        
        # Load database
        db_images = load_database(FEATURES_DB)
        
        # Generate rotated feature grids for query
        q_features_0 = query_data['features']
        q_features_90 = q_features_0.transpose(1, 0, 2)[:, ::-1, :]
        q_features_180 = q_features_0[::-1, ::-1, :]
        q_features_270 = q_features_0.transpose(1, 0, 2)[::-1, :, :]
        q_features_list = [q_features_0, q_features_90, q_features_180, q_features_270]
        
        # Perform search
        search_results = search_similar(q_features_list, db_images, top_k, workers=4)
        
        # Format results
        results = []
        conn = get_db_connection()
        
        for img_id, similarity in search_results:
            img_data = db_images[img_id]
            result_filename = os.path.basename(img_data['path'])
            
            # Get image details from main database
            image = conn.execute(
                'SELECT * FROM images WHERE filename = ?', (result_filename,)
            ).fetchone()
            
            if image:
                results.append({
                    'id': image['id'],
                    'similarity': float(similarity),
                    'filename': image['filename'],
                    'tags': image['tags'] or 'No tags yet'
                })
            else:
                # If image is in features DB but not in main DB, add basic info
                results.append({
                    'id': 0,  # Placeholder ID
                    'similarity': float(similarity),
                    'filename': result_filename,
                    'tags': 'Processing...'
                })
        
        conn.close()
        
        # Create a base64 encoded version of the query image for display
        img_str = None
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=80)
                img_str = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
        
        return render_template('results.html', 
                            results=results, 
                            query_image_data=f"data:image/jpeg;base64,{img_str}" if img_str else None,
                            query_filename=os.path.basename(image_path),
                            top_k=top_k,
                            dark_mode=dark_mode)
        
    except Exception as e:
        print(f"Error in reverse search: {str(e)}")
        flash(f'Error performing reverse search: {str(e)}', 'error')
        return redirect(url_for('reverse_search'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/tags')
def get_tags():
    """API endpoint to get popular tags"""
    conn = get_db_connection()
    tags = conn.execute(
        'SELECT name, count FROM tags ORDER BY count DESC LIMIT 50'
    ).fetchall()
    conn.close()
    
    return jsonify([dict(tag) for tag in tags])

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    dark_mode = get_dark_mode_setting()
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html', dark_mode=dark_mode)
            
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html', dark_mode=dark_mode)
            
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html', dark_mode=dark_mode)
        
        # Check if user exists
        conn = get_users_db_connection()
        existing_user = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?', 
            (username, email)
        ).fetchone()
        
        if existing_user:
            conn.close()
            flash('Username or email already exists.', 'error')
            return render_template('signup.html', dark_mode=dark_mode)
        
        # Create user
        password_hash = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        conn.commit()
        conn.close()
        
        flash('Account created successfully. Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html', dark_mode=dark_mode)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    dark_mode = get_dark_mode_setting()
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember_me = 'remember_me' in request.form
        
        # Validation
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html', dark_mode=dark_mode)
        
        # Check credentials
        conn = get_users_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        
        if not user or not check_password_hash(user['password_hash'], password):
            conn.close()
            flash('Invalid username or password.', 'error')
            return render_template('login.html', dark_mode=dark_mode)
        
        # Set session
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['is_admin'] = bool(user['is_admin'])
        
        # Set permanent session if remember me is checked
        session.permanent = remember_me
        
        conn.close()
        
        next_page = request.args.get('next')
        if next_page:
            return redirect(next_page)
            
        flash('Logged in successfully.', 'success')
        return redirect(url_for('index'))
    
    return render_template('login.html', dark_mode=dark_mode)

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page showing their uploaded images"""
    conn = get_db_connection()
    user_images = conn.execute(
        'SELECT * FROM images WHERE user_id = ? ORDER BY upload_date DESC', 
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    dark_mode = get_dark_mode_setting()
    return render_template('profile.html', images=user_images, dark_mode=dark_mode)

@app.route('/delete_image/<int:image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    """Delete an image (only allowed for owner or admin)"""
    conn = get_db_connection()
    image = conn.execute(
        'SELECT * FROM images WHERE id = ?', (image_id,)
    ).fetchone()
    
    if not image:
        conn.close()
        flash('Image not found.', 'error')
        return redirect(url_for('index'))
    
    # Check if user owns the image or is admin
    if image['user_id'] != session['user_id'] and not session.get('is_admin'):
        conn.close()
        flash('You do not have permission to delete this image.', 'error')
        return redirect(url_for('image_detail', image_id=image_id))
    
    # Delete the image file
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
    
    # Delete from database
    conn.execute('DELETE FROM images WHERE id = ?', (image_id,))
    conn.commit()
    conn.close()
    
    flash('Image deleted successfully.', 'success')
    
    # Redirect to appropriate page
    if request.referrer and 'image' in request.referrer:
        return redirect(url_for('profile'))
    else:
        return redirect(url_for('profile'))

if __name__ == '__main__':
    # Initialize databases
    init_db()
    init_features_db()
    init_users_db()  # Add this line
    
    # Start the application
    print("Starting Danbooru-like image board...")
    app.run(host='0.0.0.0', port=5000, debug=True)