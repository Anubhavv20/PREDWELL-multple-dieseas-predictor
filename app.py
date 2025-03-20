import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model  
import streamlit as st
from PIL import Image, ImageOps
# from tensorflow.keras.preprocessing import image

app = Flask(__name__)

def predict(values, dic):
    # diabetes
    # if len(values) == 8:
    #     dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
    #             'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
    #             'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}

    #     if dic['BMI'] <= 18.5:
    #         dic2['NewBMI_Underweight'] = 1
    #     elif 18.5 < dic['BMI'] <= 24.9:
    #         pass
    #     elif 24.9 < dic['BMI'] <= 29.9:
    #         dic2['NewBMI_Overweight'] = 1
    #     elif 29.9 < dic['BMI'] <= 34.9:
    #         dic2['NewBMI_Obesity 1'] = 1
    #     elif 34.9 < dic['BMI'] <= 39.9:
    #         dic2['NewBMI_Obesity 2'] = 1
    #     elif dic['BMI'] > 39.9:
    #         dic2['NewBMI_Obesity 3'] = 1

    #     if 16 <= dic['Insulin'] <= 166:
    #         dic2['NewInsulinScore_Normal'] = 1

    #     if dic['Glucose'] <= 70:
    #         dic2['NewGlucose_Low'] = 1
    #     elif 70 < dic['Glucose'] <= 99:
    #         dic2['NewGlucose_Normal'] = 1
    #     elif 99 < dic['Glucose'] <= 126:
    #         dic2['NewGlucose_Overweight'] = 1
    #     elif dic['Glucose'] > 126:
    #         dic2['NewGlucose_Secret'] = 1

    #     dic.update(dic2)
    #     values2 = list(map(float, list(dic.values())))

    #     model = pickle.load(open('models/diabetes.pkl','rb'))
    #     values = np.asarray(values2)
    #     return model.predict(values.reshape(1, -1))[0]

    # # breast_cancer
    # elif len(values) == 22:
    #     model = pickle.load(open('models/breast_cancer.pkl','rb'))
    #     values = np.asarray(values)
    #     return model.predict(values.reshape(1, -1))[0]

    # heart disease
    if len(values) == 4:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


    # liver disease
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

# @app.route("/malariapredict", methods=['POST', 'GET'])
# def malariapredictPage():
#     pred = None  # Default value for pred
#     disease_names = ['Melanoma', 'Benign Lesions of Keratosis', 'Dermatofibroma']  # Example list of disease names

#     if request.method == 'POST':
#         try:
#             img = Image.open(request.files['image'])
#             img = img.resize((224, 224))  # Resize the image
#             img.save("uploads/image.jpg")
#             img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
#             os.path.isfile(img_path)
#             img = np.array(img)  # Convert image to numpy array
#             img = np.expand_dims(img, axis=0)  # Add batch dimension

#             model = tf.keras.models.load_model("models/Cancer.h5")
#             print("Model loaded successfully")
#             pred = model.predict(img)[0]  # Assuming the model returns prediction probabilities
#             print("Prediction:", pred)
#         except Exception as e:
#             print("Error:", e)  # Print the specific error
#             message = "An error occurred. Please try again."
#             return render_template('malaria.html', message=message)
#     # return render_template('malaria_predict.html', pred=pred)
#     return render_template('malaria_predict.html', pred=pred, disease_names=disease_names)

# if __name__ == '__main__':
#     app.run(debug = True)

# @app.route("/malariapredict", methods=['POST', 'GET'])
# def malariapredictPage():
#     pred = None  # Default value for pred
#     disease_names = ['Melanoma', 'Acne', 'MonkeyPox','Eczema']  # Example list of disease names

#     if request.method == 'POST':
#         try:
#             img = Image.open(request.files['image'])
#             img = img.resize((224, 224))  # Resize the image
#             img.save("uploads/image.jpg")
#             img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
#             os.path.isfile(img_path)
#             img = np.array(img)  # Convert image to numpy array
#             img = np.expand_dims(img, axis=0)  # Add batch dimension

#             model = tf.keras.models.load_model("models/saved_model.pb")
#             print("Model loaded successfully")
#             pred = model.predict(img)[0]  # Assuming the model returns prediction probabilities
#             print("Prediction:", pred)
#         except Exception as e:
#             print("Error:", e)  # Print the specific error
#             message = "An error occurred. Please try again."
#             return render_template('malaria_predict.html', message=message)
    
#     return render_template('malaria_predict.html', pred=pred, disease_names=disease_names)

# @app.route("/malariapredict", methods=['POST', 'GET'])
# def malariapredictPage():
#     pred = None  # Default value for pred
#     # disease_names = ['acne', 'eczema', 'melanoma','monkeypox'] 
#     disease_names = ['monkeypox', 'melanoma', 'eczema','acne'] 

#     if request.method == 'POST':
#         try:
#             img = Image.open(request.files['image'])
#             img = img.resize((224, 224))  # Resize the image
#             img.save("uploads/image.jpg")
#             img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
#             os.path.isfile(img_path)
#             img = np.array(img)  # Convert image to numpy array
#             img = np.expand_dims(img, axis=0)  # Add batch dimension

#             model = tf.keras.models.load_model("models/keras_model.h5")
#             print("Model loaded successfully")
#             pred = model.predict(img)[0]  # Assuming the model returns prediction probabilities
#             print("Prediction:", pred)
#         except Exception as e:
#             print("Error:", e)  # Print the specific error
#             message = "An error occurred. Please try again."
#             return render_template('malaria_predict.html', message=message)

#     return render_template('malaria_predict.html', pred=pred, disease_names=disease_names)

# @app.route("/malariapredict", methods=['POST', 'GET'])
# def malariapredictPage():
#     pred = None  # Default value for pred
#     disease_names = ['monkeypox', 'melanoma', 'eczema','acne']  # Example list of disease names

#     if request.method == 'POST':
#         try:
#             img = Image.open(request.files['image'])
#             img = img.resize((224, 224))  # Resize the image
#             img.save("uploads/image.jpg")
#             img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
#             os.path.isfile(img_path)
#             img = np.array(img)  # Convert image to numpy array
#             img = np.expand_dims(img, axis=0)  # Add batch dimension

#             model = tf.keras.models.load_model("models/keras_model.h5")
#             print("Model loaded successfully")
#             pred = model.predict(img)[0]  # Assuming the model returns prediction probabilities
#             print("Prediction:", pred)
#         except IOError as e:
#             print("Error loading model:", e)
#             message = "An error occurred while loading the model."
#             return render_template('malaria_predict.html', message=message)
#         except Exception as e:
#             print("Error:", e)  # Print the specific error
#             message = "An error occurred. Please try again."
#             return render_template('malaria_predict.html', message=message)

#     return render_template('malaria_predict.html', pred=pred, disease_names=disease_names)

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    pred = None  # Default value for pred
    disease_names = ['MonkeyPox', 'Eczema', 'Melanoma', 'Acne']  # Updated list of disease names

    if request.method == 'POST':
        try:
            img = Image.open(request.files['image'])
            img = img.resize((224, 224))  # Resize the image
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = np.array(img)  # Convert image to numpy array
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            model = tf.keras.models.load_model("models/keras_model.h5")
            print("Model loaded successfully")

            # Debugging: Print input image shape
            print("Input image shape:", img.shape)

            pred = model.predict(img)[0]  # Assuming the model returns prediction probabilities
            print("Prediction:", pred)

            # Adjust threshold for each class
            threshold = 0.5  # Default threshold
            if pred.max() < threshold:  # If the highest probability is less than the threshold, classify as 'Other'
                pred = np.array([0, 0, 0, 1])
            else:
                pred = (pred >= threshold).astype(int)  # Apply thresholding

            # Debugging: Print the updated prediction
            print("Updated Prediction:", pred)
        except Exception as e:
            print("Error:", e)  # Print the specific error
            message = "An error occurred. Please try again."
            return render_template('malaria.html', message=message)

    return render_template('malaria_predict.html', pred=pred, disease_names=disease_names)


if __name__ == '__main__':
    app.run(debug = True)