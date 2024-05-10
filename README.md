# Image Captioning using LSTM

**Project Title:** Image Captioning using LSTM

**Technologies Used:** Python, TensorFlow, Streamlit

**Dataset:**
The project utilizes the Flickr8k dataset, a collection of images along with their corresponding captions.

**Feature Extraction using VGG16:**
Initially, the VGG16 model is employed to extract features from all images. VGG16 is a pre-trained Convolutional Neural Network (CNN) architecture typically used for tasks such as image classification and recognition. To leverage its capabilities as a feature extractor, the final layer of the VGG16 model is removed, allowing the extraction of features from the images using the intermediate layers of the model.

**Load, Clean, and Save Image Descriptions:**
The descriptions associated with the images undergo preprocessing steps to ensure uniformity and cleanliness. This includes converting text to lowercase, removing punctuation, and retaining only alphabetic characters. Subsequently, a vocabulary corpus containing all the words from the descriptions is created. The cleaned descriptions are saved for future use. The training images, descriptions, and features are loaded, and tokenization is performed on the train descriptions.

**Preparing Data for LSTM Model:**
Data preparation involves generating inputs for the LSTM model. This includes creating two inputs: one for the image features and the other for the input sequence (text) to predict the next word in the sequence (caption) as an output. Padding is applied to the input sequence with zeros to match the maximum length among all descriptions. The output sequence is then one-hot encoded with a size corresponding to the vocabulary corpus size. 

**Defining LSTM Model:**
The LSTM model architecture is defined with two input layers: one for image features and the other for the input sequence. The image features are processed through a dropout layer followed by a dense layer to extract relevant features. Similarly, the input sequence is passed through an embedding layer to convert integer-encoded words into dense vectors. A dropout layer is applied for regularization, and an LSTM layer processes the sequential input. The output of both the image and sequence models are concatenated and passed through dense layers to predict the next word in the sequence.

**Model Compilation:**
The model is compiled using categorical cross-entropy loss and the Adam optimizer.

**Training the Model:**
The model is trained for 20 epochs to iteratively improve accuracy and minimize loss. The last model iteration is considered the best model for further prediction and evaluation tasks.

**Prediction and Evaluation:**
Caption generation is performed word by word, and the generated captions are evaluated using the BLEU (Bilingual Evaluation Understudy) score. BLEU provides a numerical score representing the similarity between the machine-generated captions and reference captions. The corpus BLEU score is computed to evaluate the entire set of generated captions for test images.

**Streamlit UI:**
A user-friendly interface is developed using Streamlit, allowing users to upload images. The uploaded images undergo feature extraction using the VGG16 model. These extracted features, along with input sequences, are then fed into the LSTM model to predict caption word by word. The generated words are joined together and the caption is displayed on the interface for user interaction.

This comprehensive project description outlines the entire process from data preprocessing to model training, evaluation, and integration into a user interface using Streamlit.
