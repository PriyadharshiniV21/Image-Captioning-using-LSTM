
import tensorflow
import numpy as np
import streamlit as st
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img

# Returns a set of image_ids from the trainImages/testImages file
def load_set_of_image_ids(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    image_ids = set()
    for line in lines:
        if len(line) < 1:
            continue
        image_ids.add(line.split('.')[0])
    return image_ids

# Returns a dictionary of image_ids and their descriptions list for train/test images from the descriptions file
def load_clean_descriptions(all_desc, image_ids):
    file = open(all_desc, 'r')
    lines = file.readlines()
    descriptions = {}
    for line in lines:
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in image_ids:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

# Stores all train descriptions in a list
def to_list(descriptions):
    all_desc_list = []
    for image_id, image_desc in descriptions.items():
        for desc in image_desc:
            all_desc_list.append(desc)
    return all_desc_list

def tokenization(descriptions):
    all_desc_list = to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc_list)
    return tokenizer

# Returns maximum length (no. of words in a description) out of all the descriptions
def max_length(descriptions):
    all_desc_list = to_list(descriptions)
    return max(len(x.split()) for x in all_desc_list)

def int2word(tokenizer, integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Returns the predicted description for the given image using its features
def predict_desc(model, tokenizer, photo, max_len):
    in_seq = 'startseq'
    for i in range(max_len):
        seq = tokenizer.texts_to_sequences([in_seq])[0]
        seq = pad_sequences([seq], maxlen = max_len)
        y_hat = model.predict([photo, seq], verbose = 0)
        y_hat = np.argmax(y_hat) # Returns the index of the maximum value
        word = int2word(tokenizer, y_hat)
        if word == None:
            break
        in_seq = in_seq + ' ' + word
        if word == 'endseq':
            break
    return in_seq

def extract_features(filename):
    # Load the model
    model = VGG16()
    # Re-structure the model
    model.layers.pop()
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
    # Load the photo
    image = load_img(filename, target_size = (224, 224))
    # Convert the image pixels to a numpy array
    image = img_to_array(image)
    # Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Prepare the image for the VGG model
    image = preprocess_input(image)
    # Get features
    feature = model.predict(image, verbose = 0)
    return feature

# Load train image_ids
train_images = '/content/drive/MyDrive/ImageCaptioning/Flicker8k/Flickr_8k.trainImages.txt'
train_image_ids = load_set_of_image_ids(train_images)

# Load training descriptions
descriptions = '/content/drive/MyDrive/ImageCaptioning/descriptions.txt'
train_descriptions = load_clean_descriptions(descriptions, train_image_ids)

tokenizer = tokenization(train_descriptions)

# Pre-define the max sequence length (from training)
max_len = max_length(train_descriptions)

# Load the model
model = load_model('/content/drive/MyDrive/ImageCaptioning/best_model.h5')

st.title("Image Caption Generator")
st.markdown("Upload image to generate caption...")
image = st.file_uploader("Choose an image")
if image is not None:
    st.image(image)
    # Load and prepare the photograph
    photo = extract_features(image)
    # Generate description
    description = predict_desc(model, tokenizer, photo, max_len)
    description = ' '.join(description.split()[1:-1])
    st.write(description)
