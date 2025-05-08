import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import seaborn as sns
import os
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('mobilenet.h5')

st.title('Solar Panel Defect Detection')

option=st.selectbox('select option',options=['select','EDA','MODEL PERFORMANCE'],index=0)

data_dir='resized_dataset'

class_count={}

for class_name in os.listdir(data_dir):
    class_path= os.path.join(data_dir,class_name)
    if os.path.isdir(class_path):
        class_count[class_name]=len(os.listdir(class_path))

if option=="EDA":
    var=st.selectbox('select option',options=['','Class Dsitribution','Classes Sample images','Percentage Of Each Conditions'],index=0)
    if var =='Class Dsitribution':
        
        fig,ax=plt.subplots(figsize=(13,7))
        sns.barplot(x=list(class_count.keys()),y=list(class_count.values()),palette='viridis',ax=ax)
        ax.set_title('class distribution')
        ax.set_xlabel('class')
        ax.set_ylabel('no of image')
        ax.tick_params(rotation=45)
        st.pyplot(fig)


    if var == 'Classes Sample images':
        st.subheader("🖼️ Sample Images from Each Class")

        def show_samples(data_dir, classes, samples_per_class=2):
            fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(samples_per_class * 4, len(classes) * 3))
        
            if len(classes) == 1:
                axes = [axes]

            for row, class_name in enumerate(classes):
                class_folder = os.path.join(data_dir, class_name)
                images = os.listdir(class_folder)  # ✅ fixed this line
                chosen_images = random.sample(images, min(samples_per_class, len(images)))  # ✅ variable corrected

                for col in range(samples_per_class):
                    img_path = os.path.join(class_folder, chosen_images[col])
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    ax = axes[row][col] if len(classes) > 1 else axes[col]
                    ax.imshow(img)
                    ax.axis('off')
                    if col == 0:
                        ax.set_title(class_name, fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

        # ✅ Function call
        show_samples(data_dir, list(class_count.keys()), samples_per_class=3)

            
                    
                     
                 
                  
        

       




   



    

        
         
            
    if var =='Percentage Of Each Conditions':
        
        fig,ax=plt.subplots(figsize=(6, 6))
        ax.pie(class_count.values(), labels=class_count.keys(), autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax.set_title("📊 Percentage of Each Condition")
        st.pyplot(fig)

    
    
    
if option=='MODEL PERFORMANCE':
    
    # Mapping of labels to class names (check your class names)
     class_labels = ['Bir-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']



     uploaded_file = st.file_uploader("Choose a solar panel image...", type=["jpg", "png", "jpeg"])

     if uploaded_file is not None:
         # Load image
         img = image.load_img(uploaded_file, target_size=(224, 224))
         img_array = image.img_to_array(img) / 255.0
         img_array = np.expand_dims(img_array, axis=0)

        # Predict the image class
         prediction = model.predict(img_array)
         predicted_class = np.argmax(prediction)

         # Display the image and the prediction
         st.image(img, caption='Uploaded Image.', use_container_width=True)
         st.write(f"Prediction: {class_labels[predicted_class]}")

