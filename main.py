from importlib.resources import path
from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

st.title('Transportni klassifikatsiya qiluvchi model')

#rasmni yuklash
file = st.file_uploader('Rasm yuklash',type=['png','jpg','jpeg','gif'])
st.image(file)
#Agar Fayl mavjud bolsa keyin bularni ishlatamiz
if file:
    #PIL Convert
    img = PILImage.create(file)

    #modelni yuklaymiz
    model = load_learner('mylearn.pkl')

    #Bahorat qilish
    pred, pred_id, probs  = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)