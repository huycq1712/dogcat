import PIL
from PIL import Image
import cv2

import numpy as np
import onnxruntime as rt

import streamlit as st

import time

st.set_page_config(layout="wide")
st.title('DOG vs CAT Classifier')
st.write('This is a demo of Streamlit')

def infer_process(image, backend='onnx'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return image

def softmax(x):
  """
  This function calculates the softmax of a numpy array.
  """
  exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return exps / np.sum(exps, axis=-1, keepdims=True)

@st.cache_resource()
def load_model():
    return rt.InferenceSession('dogcat_acc_0.91.onnx', provider='CPUExecutionProvider')

label_dict = {0: 'Dog', 1: 'Cat'}

# load model
model = load_model()
# upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.write("")
    st.write("Classifying...")

    # preprocess the image
    image = image.convert('RGB')
    image = np.array(image)
    inputs = infer_process(image).astype(np.float32)

    start = time.time()
    outputs = model.run(None, {model.get_inputs()[0].name:  inputs})[0]
    rt_time = time.time() - start
    
    outputs = softmax(outputs)
    label = label_dict[np.argmax(outputs[0])]
    score = np.max(outputs[0])
    
    # a box to display the results
    st.subheader('Results')
    st.write(f'The image is a {label}')
    st.write(f'Probability of being a {label}: {score:.4f}')
    st.write(f'Inference time: {rt_time:.4f} seconds')
    
    st.subheader('Uploaded Image')
    st.image(image, caption='Uploaded Image.', use_column_width=True)


