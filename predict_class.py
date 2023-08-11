import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

model_path = 'best_model.tf' # Your model
model = load_model(model_path)

test_image_dir = r'test_images'
test_images = []

for img_name in os.listdir(test_image_dir):
    img_path = os.path.join(test_image_dir, img_name)
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    test_images.append(img_array)

test_images = np.vstack(test_images)

predictions = model.predict(test_images)

class_names = ['Meme Bold Font', "I don't know this font"] # Update list of classification if you upload more examples to train model. Example = ['Meme Bold Font', 'New class font 1', 'New class font 2', ...]
predicted_labels = [class_names[np.argmax(pred)] for pred in predictions]
print(predictions)
for img_name, label in zip(os.listdir(test_image_dir), predicted_labels):
    print(f"Checked Image: {img_name} - Predicted Font: {label}")
