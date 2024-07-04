import os
from flask import Flask, flash, request;
import PredictHeight.predict_height as height

def delete_images(id):
    os.remove(f'image{id}.png')

def find_index(): 
    cnt = 0
    while os.path.exists(f'image{cnt}.png'):
        cnt = cnt + 1
    return cnt

app = Flask(__name__)
app.secret_key = 'Tailoring App'
@app.route('/height', methods=['GET', 'POST'])

def index():
    if request.method == "POST":
        if "image" not in request.form:
            flash("Please check the parameter")
        id = find_index()
        request.files["image"].save(f"image{id}.png")
        
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