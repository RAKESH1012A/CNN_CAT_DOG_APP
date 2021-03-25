from flask import Flask, render_template, request      
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
from keras import backend as K


app = Flask(__name__)

def pred(dog_img):
    K.clear_session()
    new_model = load_model('cat_dog_100epochs.h5')
    return new_model.predict_classes(dog_img)
   

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/predict", methods=["GET","POST"])
def predict():
    
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)        
        file.save(os.path.join('static/images/', filename))
        sfname = 'static/images/'+str(secure_filename(file.filename))        
        dog_img = image.load_img(sfname, target_size=(150, 150))

        dog_img = image.img_to_array(dog_img)

        dog_img = np.expand_dims(dog_img, axis=0)
        dog_img = dog_img/255
        
        #return render_template("result.html",pred = dog_img)
        prediction_prob = pred(dog_img)
        #return render_template("result.html",pred = prediction_prob)
        a=prediction_prob.ravel()
        if a==1:
            return render_template("result.html",pred = 'Dog',pic=sfname)
        else:
            return render_template("result.html",pred = 'Cat',pic=sfname)

            
    return render_template("predict.html")

    
if __name__ == "__main__":
    app.run(debug=True)