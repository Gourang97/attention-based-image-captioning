# Attention Based Image Captioning System [NLP Application]

<br/><center><img src="https://github.com/Gourang97/Fashion_dataset_uml/blob/master/Results/FirstSlide.PNG" width="800" height="400"></center>

# Author - Gourang Patel [www.linkedin.com/in/gourang-patel]

## Idea

<p align="justify">We are trying to encounter the problem faced by the "classic" image captioning method using the
Attention mechanism in the decoder. The attention mechanism will help the decoder to focus
on relevant parts of the image. The decoder will only use specific parts of the image rather than
conditioning on the entire hidden state h produced from the convolutional neural network.We can observe in the figure <br/><center><img src="https://github.com/Gourang97/Fashion_dataset_uml/blob/master/Results/Dataset_img.PNG" width="800" height="400"></center></p>, there is one additional layer from the classic architecture, and this new
layer makes the model as attention model. While predicting the next word while generating captions
for an image, if we have previously predicted i words, the hidden state will be hi. Then the model
will select the relevant part of the image using the attention mechanism which will be zi (which
captures only relevant information from the image) and this will be go as an input to the LSTM. The
LSTM then generates the next word and also passes on the information to the new hidden state hi+1
The model architecture is inspired by the Show, Attend and Tell paper, where an attention
based model was introduced to automatically learn and describe the content of images.</p>

## Problem Statement

<p align="justify">Automatically describing the content of an image is a fundamental problem in artificial intelligence
that connects computer vision and natural language processing. Some of its application includes
helping visually impaired people better understand the content of images on the web.
One of the key challenge involves generating description that must capture not only the objects
contained in an image, but also express how these objects relate to each other as well as their
attributes and the activities they are involved in. Moreover, the above semantic knowledge has to
be expressed in a natural language like English, which means that a language model is needed in
addition to visual understanding.</p>

## Dataset 
<p align="justify">We leveraged Flickr 8K dataset consisting 5 captions for each image to train our model. The dataset
contains a total of 8092 images each with 5 captions, in total we have 40460 properly labelled
captions.</p>
<br/><center><img src="https://github.com/Gourang97/Fashion_dataset_uml/blob/master/Results/FirstSlide.PNG" width="800" height="400"></center>

## Dataset Preprocessing
- We conducted following data Preprocessing steps
• Cleaned the captions by removing punctuations, single characters, and numeric values.
• Added start and end tags for every caption, so that model understands the start and end of
each caption.
• Resized images to 224 X 224 followed by pixel normalization to suit our VGG16 image
feature extraction model.
• Tokenized the captions (for example, by splitting on spaces) to obtain a vocabulary of unique
words in the data.
• Padded all the sequence to be the same length as the longest one.
## Results
