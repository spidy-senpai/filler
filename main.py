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
        # Fix: Use square brackets
        if 'image' not in request.files:
            return {'error': 'No image file provided'}, 400
        
        image_file = request.files['image']
        
        # Process with your model
        colorized_image = fillcolour_model(image_file)
        
        # Convert to bytes and return
        img_byte_arr = io.BytesIO()
        colorized_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        import base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return {
            'image': f'data:image/jpeg;base64,{img_base64}'
        }
    
        # return send_file(
        #     img_byte_arr,
        #     mimetype='image/jprg',
        #     as_attachment=False
        # )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)
