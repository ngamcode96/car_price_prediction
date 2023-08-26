import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 

#encoder
from category_encoders import OrdinalEncoder

from services import load_data

st.title("Car price prediction")

data = load_data("./saved/categorial_data/data.json")
manufacturerOptions = load_data("./saved/categorial_data/manufacturer.json")
colorOptions = load_data("./saved/categorial_data/color.json")
doorsOptions = load_data("./saved/categorial_data/doors.json")
drive_wheelsOptions = load_data("./saved/categorial_data/drive_wheels.json")
fuel_typeOptions = load_data("./saved/categorial_data/fuel_type.json")
gear_box_typeOptions = load_data("./saved/categorial_data/gear_box_type.json")
leather_interiorOptions = load_data("./saved/categorial_data/leather_interior.json")
wheelOptions = load_data("./saved/categorial_data/wheel.json")
cylindersOptions = load_data("./saved/categorial_data/cylinders.json")
engine_volumeOptions = load_data("./saved/categorial_data/engine_volume.json")
airbagsOptions = load_data("./saved/categorial_data/airbags.json")



predict_price = -1

def bool_to_int(x):
    if (x == "True"):
        return 1
    else:
        return 0

def prediction_process():
    car = np.array([[levy, manufacturer,model, year, category, leather_interior, fuel_type, engine_volume, milage, cylinders, gear_box_type, drive_wheels, doors, wheels, color, airbags, has_turbo]])
    columns_names = load_data("./saved/columns.json")
    X = pd.DataFrame(car, columns=columns_names)

    X["levy"] = X["levy"].astype("float32")
    X["year"] = X["year"].astype("int32")
    X["engine_volume"] = X["engine_volume"].astype("float32")
    X["mileage"] = X["mileage"].astype("float32")
    X["cylinders"] = X["cylinders"].astype("float32")
    X["airbags"] = X["airbags"].astype("float32")
    X["turbo"] = X["turbo"].apply(bool_to_int)


    
    num_cols = X.select_dtypes('number').columns.to_list()
    cat_cols = X.select_dtypes('object').columns.to_list()

    # print(cat_cols)
    # print(num_cols)

    #encoding categorical features
    with open("./saved/encoder.pkl", "rb") as encoder_file:
        ordinalEncoder = joblib.load(encoder_file)
        X[cat_cols] = ordinalEncoder.transform(X[cat_cols])


    #scaler
    with open("./saved/standard_scaler.pkl", "rb") as sc_file:
        sc = joblib.load(sc_file)
        X[num_cols] = sc.transform(X[num_cols])

    
    #decompress model 
    if not os.path.exists("./saved/final_model.pkl"):
        import gzip
        with gzip.open('./saved/final_model.pkl.gz', 'rb') as compressed_file:
            with open('./saved/final_model.pkl', 'wb') as pkl_file:
                pkl_file.writelines(compressed_file)
    
    #load model
    with open("./saved/final_model.pkl", "rb") as model_file:
        final_model = joblib.load(model_file)

        price = final_model.predict(X)
        price = round(price[0])
    
    return price
    

col1, col2, col3 = st.columns(3)

with col1:
    manufacturer = st.selectbox("Manufacturer: ", options=manufacturerOptions)
    model = st.selectbox("Model: ", options=data[manufacturer]["model"])
    category = st.selectbox("Category: ", options=data[manufacturer]["category"])
    fuel_type = st.selectbox("Fuel type: ", options=fuel_typeOptions)
    gear_box_type = st.selectbox("Gear box type: ", options=gear_box_typeOptions)
    leather_interior = st.selectbox("Leather interior: ", options=leather_interiorOptions)


with col2:
    doors = st.selectbox("Doors: ", options=doorsOptions)
    drive_wheels = st.selectbox("Drive wheels: ", options=drive_wheelsOptions)
    wheels = st.selectbox("Wheels: ", options=wheelOptions)
    color = st.selectbox("Color: ", options=colorOptions)
    year = st.text_input("Year production: (1940 - 2015)", placeholder="2010", value=2010)
    levy = st.text_input("Levy: ", value=1328)


with col3:
    milage = st.text_input("Mileage: ", placeholder="(km)", value=186300)
    engine_volume = st.selectbox("Engine volume", options=engine_volumeOptions)
    has_turbo = st.selectbox("Has turbo:", options=[False, True])
    cylinders = st.selectbox("Cylinders: ", options=cylindersOptions)
    airbags = st.selectbox("Airbags: ", options=airbagsOptions)


btn_clicked = st.button("Predict price", type="primary")

if(btn_clicked):
    with st.spinner("Loading..."):
        predict_price = prediction_process()

    st.subheader("Price : " + str(predict_price) + " $")


