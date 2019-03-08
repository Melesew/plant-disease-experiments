import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import subprocess
from keras.applications.inception_v3 import preprocess_input
import argparse

parser = argparse.ArgumentParser()

#constants
# model_path = os.path.join('../Plant_Disease_Detection_Benchmark_models/Models', 'VGG_scratch_94.h5') # using vgg net model
# model_path = os.path.join('../Plant_Disease_Detection_Benchmark_models/Models', 'ResNet_plant_97.h5') # using resnet model
# model_path = os.path.join('../Plant_Disease_Detection_Benchmark_models/Models', 'Inception_Scratch_95.h5') # using Inception Scratch model
model_path = os.path.join('../Plant_Disease_Detection_Benchmark_models/Models', 'InceptionV3-plantDataset-noAugmentation_70.h5') # using resnet model

model = load_model(model_path)

data_path = "/home/icog/Mele/generated_data/"


target_size = (100, 100)

def predict(img_path):
    array = [0] * 38
    for f in os.listdir(img_path):
        if f.startswith('.'): #path all .gits
            continue
        img = Image.open(img_path + f)
        if img.size != target_size:
            img = img.resize(target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        label = np.argmax(preds)
        result = str(label)
        
        idx = int(result)
        array[idx] += 1

    return array

if __name__ == "__main__":
    file = open("confussion_matrix_4_generated_inceptionV3.txt", "a")
    image_path = os.listdir(data_path)
    image_path.sort()
    for path in image_path:
        if path.startswith('.'): # to pass .git folder
            continue
        matrix_array = predict(data_path+path+'/')
        print(matrix_array)
        file.write(str(matrix_array) + '\n')
    file.close()
        
    
