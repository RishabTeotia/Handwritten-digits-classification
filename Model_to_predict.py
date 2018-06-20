# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:46:02 2018

@author: Rishab Teotia
"""

#import tensorflow as tf
#import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from Digits_classify_Handwritten import test_images,test_labels
my_model=load_model('ModelHandwritten.h5')

# This is used to prediction for whole data 
i=0
for i in range (10):
        predictions=my_model.predict_classes(test_images[:])
        print(predictions)
        
# This prediction is for particular data
pred=my_model.predict_classes(test_images[5:8])
print(pred)
plt.imshow(test_labels[5:8])
loss, accuracy = my_model.evaluate(test_images, test_labels)
print('Test accuracy: %.2f' % (accuracy))
