#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import shutil
import os
import pickle
from src.components.load_data import load
from src.components.generate_image import generate
from src.components.train_model import Generator, train
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__, static_folder="C:\\Users\\jerry\\Downloads\\Personal_Projects\\Image_Generation\\templates\\static")

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/train', methods=['POST'])
def train_model():
    val=request.form['val']
    file=request.files['file']
    file.save('artifacts/tmp/data.pkl')
    gen={}
    with open('artifacts/tmp/data.pkl', 'rb') as f:
        data = pickle.load(f)
    for i in data:
        gen[i] = train(data[i],iters=int(val))
    with open('artifacts/tmp/model.pkl', 'wb') as f:
        pickle.dump(gen, f)
    return redirect(url_for("index"))

@app.route('/generate', methods=['POST'])
def generate_image():
    val = request.form['val']
    if os.path.exists('artifacts/tmp/model.pkl'):
        with open('artifacts/tmp/model.pkl', 'rb') as f:
            gen = pickle.load(f)
    else:
        print("use default")
        with open('artifacts/model.pkl', 'rb') as f:
            gen = pickle.load(f)
    generate(val,gen,"templates/static/")
    
    return render_template('index.html',filename="fig.jpg")


@app.route('/display/<filename>')
def display(filename):
    return redirect(url_for('static', filename=filename))
 
if __name__ == "__main__":
    app.run(debug=True)