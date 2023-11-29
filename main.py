import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np
# import sklearn

class_list = {'0' : 'NORMAL', '1' : 'Pneumonia'}

st.title('Pneu Detection')

with open('lrc_xray.pkl', 'rb') as file:
  model = pkl.load(file)

st.header('up image')
image = st.file_uploader('choose!', type = (['png', 'jpg', 'jpeg']))

if image is not None:
  image = Image.open(image)
  st.image(image, caption='test iamge')

  if st.button('Predict'):
    image = image.resize((227*227*3, 1))
    feature_vector = np.array(image)
    label = str(model.predict(feature_vector)[0])

    st.header('Result')
    st.text(class_list[label])
    
