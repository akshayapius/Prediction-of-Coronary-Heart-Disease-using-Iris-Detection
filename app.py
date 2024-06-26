from flask import Flask, render_template, request, flash, redirect
import os
from predict import predict

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('design.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_image', methods=['POST'])
def submit_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        img = request.files['image']
        # If the user does not select a file, the browser submits an empty file without a filename
        if img.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if img:
            # Define the upload folder
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)  # Ensure directory exists
            # Save the uploaded image
            base_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(base_dir, upload_folder, 'uploaded_image.jpg')
            img.save(img_path)
            # Call the predict function with the path of the uploaded image
            res = predict(img_path)
            return render_template('design.html', res=res, image_path=img_path)

if __name__ == '__main__':
    app.run()

