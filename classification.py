import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache
def load_iris():
    iris=load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species']=iris.target
    return df, iris.target_names

df, target_names = load_iris()
model=RandomForestClassifier()
model.fit(df.iloc[:,:,-1], df['species'])
st.sidebar.title("input Feature")
sepal_length=st.sidebar.slider("sepal length", 4.3, 7.9, 5.1)
sepal_width=st.sidebar.slider("sepal width", 2.0, 4.4, 3.5)
petal_length=st.sidebar.slider("petal length", 1.0, 6.9, 1.9)
petal_width=st.sidebar.slider("petal width", 0.1, 2.5, 0.2)

input_data=[[sepal_length, sepal_width, petal_length, petal_width]]
prediction=model.predict(input_data)
predicted_species=target_names[prediction[0]]

st.write("Prediction")
st.write(predicted_species)

