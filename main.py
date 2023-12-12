import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np
# import sklearn

class_list = {'0' : 'NORMAL', '1' : 'Pneumonia'}

st.title('Pneumonia Detection')

image_url = 'https://th.bing.com/th/id/R.5cf88f668fe816d63fb3a2464f40072d?rik=pXI1kUljyoblKQ&pid=ImgRaw&r=0'
st.image(image_url, caption='Do you find hard breathing?')


input = open('lrc_xray.pkl', 'rb')
model = pkl.load(input)

st.header('Upload your image')
image = st.file_uploader('choose!', type = (['png', 'jpg', 'jpeg']))

if image is not None:
  image = Image.open(image)
  st.image(image, caption='test image')

  if st.button('Predict'):
    image = image.resize((227*227*3, 1))
    feature_vector = np.array(image)
    label = str((model.predict(feature_vector))[0])

    st.header('Result')
    st.text(class_list[label])
    
