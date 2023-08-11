import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def load_data_for_font(font_dir, font_label):
    images = []
    labels = []

    for img_name in os.listdir(font_dir):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(font_dir, img_name), target_size=(64, 64)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(font_label)

    return images, labels

font_dirs = [
    (r"dataset/bold_font", "Bold Meme Font"),
    # add diffrent dirs. For example (r"DataSet/article_font", "Article Font") (r"directory/path", "Classification of this font")
]

num_classes = 2 # Change value if you update font_dirs. New value in font_dirs = + 2 to num_clasess

all_train_images = []
all_train_labels = []

for font_dir, font_label in font_dirs:
    images, labels = load_data_for_font(font_dir, font_label)
    all_train_images.extend(images)
    all_train_labels.extend(labels)

train_images, test_images, train_labels, test_labels = train_test_split(
    all_train_images, all_train_labels, test_size=0.2, random_state=42
)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

train_labels_onehot = to_categorical(train_labels_encoded, num_classes=num_classes)
test_labels_onehot = to_categorical(test_labels_encoded, num_classes=num_classes)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    np.array(train_images),
    train_labels_onehot,
    batch_size=32,
    shuffle=True  # Set shuffle to True or False. True for best results
)

# CNN Model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(num_classes, activation='softmax'))

# If you want faster - change new_lr to higher value. This value is optimal to get best result
new_lr = 0.000001
optimizer = Adam(learning_rate=new_lr)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('best_model.tf', save_best_only=True, save_weights_only=False)

history = model.fit(
    train_generator,
    epochs=100,  # Change this value if you have small dataset. Small DataSet - higher value
    validation_data=(np.array(test_images), test_labels_onehot),
    shuffle=False,
    callbacks=[model_checkpoint]
)
model.save("font_model.tf")