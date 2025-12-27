from flask import Flask, request, send_file, render_template
import io
from flask_cors import CORS
from PIL import Image
from colorize import fillcolour_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('home.html')

@app.route("/api/colorize", methods=['POST'])
def colorize_image():
    try:
        if 'image' not in request.files:
            return {'error': 'No image file provided'}, 400
        
        image_file = request.files['image']
        print(f"Processing image: {image_file.filename}")
        
        # Process with your model
        colorized_image = fillcolour_model(image_file)
        print("Image colorized successfully")
        
        # Convert to bytes and return as base64
        img_byte_arr = io.BytesIO()
        colorized_image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        
        import base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        print(f"Base64 encoded, size: {len(img_base64)} bytes")
        
        # Return with proper JSON structure
        response = {
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}'
        }
        
        return response, 200
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR in colorize_image:\n{error_msg}")
        return {
            'success': False,
            'error': str(e)
        }, 500

if __name__ == "__main__":
    app.run(debug=True)

# Expose app for Vercel
# This line is needed for Vercel to find and serve the Flask app
