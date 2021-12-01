from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)
data=pd.read_csv("Clean_data1.csv")
pipe = pickle.load(open("LR1.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['County'].unique())
    type = sorted(data['Prop_type'].unique())
    return render_template('index.html', locations=locations, type=type)

@app.route('/predict', methods=['POST'])
def predict():
    Prop_type = request.form.get('type')
    Area = int(request.form.get('area'))
    Bedrooms = int(request.form.get('bed'))
    Bathrooms = int(request.form.get('bath'))
    Receptions = int(request.form.get('Reception'))
    County = request.form.get('location')

    print(Prop_type, Area, Bedrooms, Bathrooms, Receptions, County)
    input = pd.DataFrame([[Prop_type,Area,Bedrooms,Bathrooms,Receptions,County]],columns=['Prop_type', 'Area', 'Bedrooms', 'Bathrooms', 'Receptions', 'County'])
    prediction1 = pipe.predict(input)[0]
    return str(np.round(prediction1,2))




if __name__== "__main__":
    app.run(debug=True, port=5000)

