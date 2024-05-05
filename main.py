import streamlit as st
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the InceptionV3 model pre-trained on ImageNet data
model = InceptionV3(weights='imagenet')

# Define calorie counts for fruits and vegetables
calories_dict = {
    'apple': 52,
    'banana': 89,
    'orange': 62,
    'strawberry': 32,
    'broccoli': 31,
    'carrot': 41,
    'tomato': 22,
    # Add more items as needed
}


# Function to predict the food item in the image
def predict_food(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Get only the top prediction

    return decoded_predictions[0][1].lower()  # Return the label of the top prediction in lowercase


# Streamlit UI
st.title("Food Image Classifier")

# File uploader widget to allow users to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Perform prediction if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(299, 299))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to an image and perform prediction
    food_name = predict_food(img)

    # Display predicted food item
    if food_name:
        st.subheader("Predicted Food Item:")
        st.write(food_name.capitalize())

        # Get calorie count for predicted food item
        calories = calories_dict.get(food_name, "Calorie count not available")
        st.subheader("Calorie Count:")
        st.write(calories)
