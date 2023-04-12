from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

MODEL = tensorflow.keras.models.load_model(os.path.join(MODEL_DIR, 'RESNET50_PLANT_DISEASE.h5'))
REC_MODEL = pickle.load(open(os.path.join(MODEL_DIR, 'RF.pkl'), 'rb'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust', 'Corn(maize) Northern Leaf Blight', 'Corn(maize) healthy', 'Grape Black rot', 'Grape Esca(Black Measles)', 'Grape Leaf blight(Isariopsis Leaf Spot)', 'Grape healthy', 'Orange Haunglongbing(Citrus greening)', 'Peach Bacterial spot', 'Peach healthy', 'Bell PepperBacterial_spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites (Two-spotted spider mite)', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

@app.route('/')
def home():
        return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/plantdisease/<filenames>/<results>')
def plantresult(filenames, results):
    filenames = filenames.split(',')
    results = results.split(',')
    corrected_results = []
    for res in results:
        corrected_results.append(res.replace('_', ' ').replace(",", "").replace("'", "").replace("[", "").replace("]", ""))
    img_urls = []
    for filename in filenames:
        img_url = url_for('send_uploaded_file', filename=filename.replace(" ", "").replace(",", "").replace("'", "").replace("[", "").replace("]", ""))
        img_urls.append(img_url)
    return render_template('plantdiseaseresult.html', corrected_results=corrected_results, img_urls=img_urls)


@app.route('/disease-details/<disease_name>')
def disease_details(disease_name):
    img_url = request.args.get('img_url')
    return render_template('diseasedetails.html', disease_name=disease_name, img_url=img_url)


@app.route('/plantdisease', methods=['GET', 'POST'])
def plantdisease():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file')
        filenames = []
        results = []
        for file in files:
            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)
                model = MODEL
                imagefile = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                imagefile = imagefile.resize((224, 224))
                input_arr = tensorflow.keras.preprocessing.image.img_to_array(imagefile)
                input_arr = np.array([input_arr])
                result = model.predict(input_arr)
                probability_model = tensorflow.keras.Sequential([model, 
                                             tensorflow.keras.layers.Softmax()])
                predict = probability_model.predict(input_arr)
                p = np.argmax(predict[0])
                res = CLASSES[p]
                results.append(res)

        return redirect(url_for('plantresult', filenames=filenames, results=results))
    return render_template("plantdisease.html")


@app.route('/bucket/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__== "__main__":
    app.run(debug=True)
