import cv2
import streamlit as st
import numpy as np 
from PIL import Image


st.write("""
          # 數位藝術 Digital Art App!

          """
          )

st.write("This is an app to turn your Photos into Art")
#file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

#st.title("Lung Ultrasound Image Classification")
#st.header("Covid ?")
st.text("Upload your favorable photo to turn into a painting")


file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])  
option = st.sidebar.selectbox('Which digital transform would you like to apply?',("Oil Painting", "Water Painting", "Pencil Sketch","Color Pencil Sketch"))


def add_parameter_ui(transform_name):
    params = dict()
    if transform_name == "Oil Painting":
      o1 = st.sidebar.number_input('neighbouring = (2*size+1), input size (integer)', 5)
      params["o1"] = o1
      o2 = st.sidebar.number_input('image is divided by dynRatio before histogram processing, input dynRatio(integer)', 1)
      params["o2"] = o2
    elif transform_name == "Water Painting":
      w1 = st.sidebar.slider("w1 (default=20)", 1, 200, 20)
      params["w1"] = w1
      w2 = st.sidebar.slider("w2 (default=0.5)", 0.01, 1.0, 0.5)
      params["w2"] = w2
    else:
      p1 = st.sidebar.slider("p1 (default=60)", 1, 200,60)
      params["p1"] = p1
      p2 = st.sidebar.slider("p2 (default=0.07)", 0.01, 1.0,0.07)
      params["p2"] = p2
      p3 = st.sidebar.slider("p3 (default=0.05)", 0.01, 0.1,0.05)
      params["p3"] = p3
    return params

params = add_parameter_ui(option)

	
if file is None:
    st.text("You haven't uploaded an image file")
elif option == "Oil Painting":
    image = Image.open(file)
    img = np.array(image)  
    st.text("Your original image")
    st.image(image, use_column_width=True)
    st.text("Your digital art image")
    oil = cv2.xphoto.oilPainting(img,size=params["o1"],dynRatio=params["o2"]) 
    st.image(oil, use_column_width=True)
elif option == "Water Painting":
    image = Image.open(file)
    img = np.array(image)    
    st.text("Your original image")
    st.image(image, use_column_width=True)
    st.text("Your digital art image")
    wat = cv2.stylization(img,sigma_s=params["w1"],sigma_r=params["w2"])
    st.image(wat, use_column_width=True)
elif option == "Pencil Sketch":
    image = Image.open(file)
    img = np.array(image)   
    st.text("Your original image")
    st.image(image, use_column_width=True)
    st.text("Your digital art image")
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=params["p1"],sigma_r=params["p2"],shade_factor=params["p3"])
    st.image(dst_gray, use_column_width=True)
else:
    image = Image.open(file)
    img = np.array(image)   
    st.text("Your original image")
    st.image(image, use_column_width=True)
    st.text("Your digital art image")
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=params["p1"],sigma_r=params["p2"],shade_factor=params["p3"])
    st.image(dst_color, use_column_width=True)
	
