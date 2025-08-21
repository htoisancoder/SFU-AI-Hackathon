import argparse
import os
import io
import base64

import flask
from flask import Flask, request, jsonify, render_template
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

TITLE = "WaifuDiffusion Tagger"
DESCRIPTION = """
Demo for the WaifuDiffusion tagger models
"""

VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    return parser.parse_args()


def load_labels(dataframe) -> list[str]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

class Predictor:
    def __init__(self):
     self.model_target_size = None
     self.last_loaded_repo = None
     self.model = None
     self.providers = self.get_providers()
     print(f"Available ONNX Runtime providers: {self.providers}")
     print(f"Using providers: {self.providers[0]}")
     
     # Preload the default model
     print("Preloading model...")
     self.load_model(VIT_LARGE_MODEL_DSV3_REPO)
     print("Model preloaded successfully")

    def get_providers(self):
        """Get available execution providers with GPU priority"""
        available_providers = rt.get_available_providers()
        preferred_order = [
            'CUDAExecutionProvider', 
            'DmlExecutionProvider',  # For DirectML (Windows)
            'TensorrtExecutionProvider',
            'CPUExecutionProvider'
        ]

        ordered_providers = []
        for provider in preferred_order:
            if provider in available_providers:
                ordered_providers.append(provider)
        
        return ordered_providers or available_providers

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
        )
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        del self.model
        
        # Create session options for better performance
        session_options = rt.SessionOptions()
        session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        
        # Create ONNX Runtime session with GPU support
        model = rt.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=self.providers
        )
        
        # Print provider info for debugging
        print(f"Model loaded with provider: {model.get_providers()[0]}")
        
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image,
        model_repo=VIT_LARGE_MODEL_DSV3_REPO,
        general_thresh=0.35,
        character_thresh=0.85,
    ):
        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)
        top_rating = max(rating.items(), key=lambda x: x[1])[0] if rating else ""

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", r"\(").replace(")", r"\)")
        )

        return {
             "tags_string": sorted_general_strings,
             "rating": top_rating,  # Add this line
}

# Initialize the predictor
predictor = Predictor()

@app.route('/tag', methods=['POST'])
def tag_image():
    try:
        # Get parameters from request
        model_repo = request.form.get('model_repo', VIT_LARGE_MODEL_DSV3_REPO)
        general_thresh = float(request.form.get('general_thresh', 0.35))
        character_thresh = float(request.form.get('character_thresh', 0.85))
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGBA")
        
        # Run prediction
        result = predictor.predict(
            image=image,
            model_repo=model_repo,
            general_thresh=general_thresh,
            character_thresh=character_thresh
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

# Add this route to app.py after the existing routes
@app.route('/image/<int:image_id>')
def image_detail(image_id):
    """Display detailed view of a single image"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error.', 'danger')
        return redirect(url_for('index'))
    
    # Get image metadata
    image = conn.execute(
        'SELECT * FROM images WHERE id = ?', (image_id,)
    ).fetchone()
    
    if not image:
        flash('Image not found.', 'danger')
        conn.close()
        return redirect(url_for('index'))
    
    # Get similar images using reverse image search
    similar_images = []
    try:
        # Load features database
        features_conn = sqlite3.connect('image_features.db')
        features_conn.row_factory = sqlite3.Row
        features_cursor = features_conn.cursor()
        
        # Find the image in features database
        features_cursor.execute(
            "SELECT id FROM images WHERE path LIKE ?", 
            (f'%{image["filename"]}',)
        )
        features_row = features_cursor.fetchone()
        
        if features_row:
            features_id = features_row['id']
            
            # Get features for this image
            features_cursor.execute(
                "SELECT grid_row, grid_col, hue, saturation, value FROM chunks WHERE image_id = ?",
                (features_id,)
            )
            chunks = features_cursor.fetchall()
            
            if chunks:
                # Reconstruct features array
                features = np.zeros((GRID_SIZE, GRID_SIZE, 3))
                for chunk in chunks:
                    row, col, h, s, v = chunk
                    features[row, col] = [h, s, v]
                
                # Generate rotated versions for search
                features_0 = features
                features_90 = features_0.transpose(1, 0, 2)[:, ::-1, :]
                features_180 = features_0[::-1, ::-1, :]
                features_270 = features_0.transpose(1, 0, 2)[::-1, :, :]
                features_list = [features_0, features_90, features_180, features_270]
                
                # Load all database images
                db_images = {}
                features_cursor.execute("SELECT id, path FROM images")
                for row in features_cursor.fetchall():
                    db_images[row['id']] = {
                        'path': row['path'],
                        'features': np.zeros((GRID_SIZE, GRID_SIZE, 3))
                    }
                
                # Load all chunks
                features_cursor.execute("SELECT image_id, grid_row, grid_col, hue, saturation, value FROM chunks")
                for row in features_cursor.fetchall():
                    img_id, row_idx, col_idx, h, s, v = row
                    db_images[img_id]['features'][row_idx, col_idx] = [h, s, v]
                
                # Search for similar images (excluding the current one)
                search_results = search_similar(features_list, db_images, top_k=6, workers=2)
                
                # Get metadata for similar images
                for img_id, similarity in search_results:
                    if img_id != features_id and similarity > 0.3:  # Only show reasonably similar images
                        img_path = db_images[img_id]['path']
                        filename = os.path.basename(img_path)
                        
                        # Get metadata from main database
                        meta = conn.execute(
                            'SELECT * FROM images WHERE filename = ?', (filename,)
                        ).fetchone()
                        
                        if meta:
                            similar_images.append({
                                'id': meta['id'],
                                'filename': meta['filename'],
                                'similarity': similarity,
                                'tags': meta['tags']
                            })
                
                # Sort by similarity
                similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        
        features_conn.close()
        
    except Exception as e:
        logger.error(f"Error finding similar images: {str(e)}")
        # Continue without similar images
    
    conn.close()
    
    return render_template('image_detail.html', 
                         image=image, 
                         similar_images=similar_images[:5])  # Limit to 5 similar images

def main():
    args = parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False
        )

if __name__ == "__main__":
    main()

    # Line 86 is where the preloading 