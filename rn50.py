from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np


model = ResNet50(weights='imagenet')


img_path = 'example.jpg'  
img = image.load_img(img_path, target_size=(224, 224)) 
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) 
x = preprocess_input(x)  


predictions = model.predict(x)
print('Predicted:', decode_predictions(predictions, top=3)[0])  
