# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from classifier import classifyImage
from secrets import token_hex

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            country, country_conf, city, city_conf, top_countries = classifyImage(filepath)
            
            os.remove(filepath)
            
            return render_template('results.html', 
                                 country=country, 
                                 city=city,
                                 filename=original_filename,
                                 country_conf=country_conf,
                                 city_conf=city_conf,
                                 top_countries=top_countries)
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload an image.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1234, debug=True)
