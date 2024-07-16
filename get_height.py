import os
from flask import Flask, flash, request;
import PredictHeight.predict_height as height

from PIL import Image
import pillow_heif
import io

def delete_images(id):
    os.remove(f'image{id}.png')

def find_index(): 
    cnt = 0
    while os.path.exists(f'image{cnt}.png'):
        cnt = cnt + 1
    return cnt

def heif2png(file):
    # Read the image file
    heif_file = pillow_heif.read_heif(io.BytesIO(file.read()))
    # Convert HEIC to a format that PIL can handle
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return image

app = Flask(__name__)
app.secret_key = 'Tailoring App'
@app.route('/height', methods=['GET', 'POST'])

def index():
    if request.method == "POST":
        if "image" not in request.form:
            flash("Please check the parameter")
        id = find_index()
        img = request.files["image"]
        try:
            img.save(f"image{id}.png")
        except:
            heif2png(img).save(f'side{id}.png', format="PNG")
        
        if request.files["image"].name == "":
            delete_images(id)
            flash("Check the image if it is valid")
    else:
        flash("Should use POST method")
        return
            
    _height = height.predict(f'image{id}.png')
    
    delete_images(id)
    return str(_height)
    
if __name__ == "__main__":
    height.load_model()
    app.run(host = "0.0.0.0", port = 8001)