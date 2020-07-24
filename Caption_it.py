#!/usr/bin/env python
# coding: utf-8

# In[100]:


# standalone notebook to generate caption for a complete new random image
#since model is loaded for prediction and the model will be needing encoding features of the image and start seq of the caption to start
# so for preprocessing and encoding image -preprocess_img and encode_img are written
# so we get image features as encoded_img as (1,2048) dim then we need two dicts word_to_inde and index_to_word
# becuse the model will predict prob dist for all vocab words so the index of the word mapped to word will ne needed


# In[101]:


## data cleaning
import json
from time import time
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input , decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense , LSTM , Input , Dropout , Embedding
from keras.layers.merge import add


# In[102]:


import numpy as np


# In[103]:


#load model
model = load_model('model_9.h5')
model._make_predict_function()


# In[104]:


model_temp = ResNet50(weights='imagenet',input_shape=(224,224,3))


# In[105]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
model_resnet._make_predict_function()

# In[106]:


def preprocess_img(img): 
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    #expand_dims or reshape can be used so that img_size (batch_size,224,224,3) cause that is the input_shape expected by resnt
    img = np.expand_dims(img,axis=0)
    #now img_size = (1,224,224,3)
    #normalisation - now each network has some specific type of preprocessing so for resnet we have imported preprocess_input
    #to serve that purpose
    img = preprocess_input(img)
    return img


# In[125]:


# method to encode img into image_features of shape 2048
def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    #feature_vector's shape = 1,2048 so convert back into (2048,)
    feature_vector = feature_vector.reshape((1,feature_vector.shape[1]))
    #print(feature_vector.shape)
    return feature_vector


# In[126]:


with open("word_to_index.pkl" , "rb") as w2i:
    word_to_index = pickle.load(w2i)


# In[127]:


word_to_index


# In[128]:


index_to_word = {}


# In[129]:


for word,idx in word_to_index.items():
    index_to_word[idx] = word


# In[130]:


index_to_word


# In[135]:


#Prediction 
def predict_caption(photo):
    in_text = 'startseq'
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence],maxlen = max_len,padding='post')#it take 2d array thts y [sequence]
        
        y_pred = model.predict([photo,sequence])
        y_pred = y_pred.argmax()
        #print(y_pred)
        word = index_to_word[y_pred]
        in_text += (' '+word)
        
        if word == 'endseq':
            break
            
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption
    


# In[136]:


#enc = encode_img("IMG_20181005_211317_050.jpg")


# In[137]:


#predict_caption(enc)


# In[138]:


def caption_this_img(img):
    enc = encode_img(img)
    caption = predict_caption(enc)
    return caption


# In[139]:


#caption_this_img("IMG_20170209_121146.jpg")


# In[ ]:




