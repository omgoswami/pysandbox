#no answer key for test data
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#getting training data from the laptop
train_dir = "/Users/omgoswami/Downloads/dogs-vs-cats/train"
train_files = os.listdir(train_dir)

#getting test data from the laptop
test_dir = "/Users/omgoswami/Downloads/dogs-vs-cats/test1"
test_files = os.listdir(test_dir)

#making the labels for the training data
categories = []
for filename in train_files:
    if 'cat' in filename:
        categories.append('cat')
    else:
        categories.append('dog')
labels = pd.DataFrame({
    'filename' : train_files,
    'category' : categories
})

#make the labels for test data
test_categories = []
for filename in test_files:
    if 'cat' in filename:
        test_categories.append('cat')
    else:
        test_categories.append('dog')

test_df = pd.DataFrame({
    'filename': test_files,
    'category': test_categories
})
#preparing data
#split training data into training and validation
train_df, validation_df = train_test_split(labels, test_size=0.25, random_state=42)
#reset indices of training data
train_df.reset_index(drop=True)
#reset indices of validation data
validation_df.reset_index(drop=True)
#getting total image counts
total_train = train_df.shape[0]
total_validation = validation_df.shape[0]
total_test = test_df.shape[0]
batch_size = 15
image_width = 128
image_height = 128
image_size = (image_width, image_height)
image_channels = 3
num_classes = 2

#preparing training and validation data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    train_dir,
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_generator = validation_datagen.flow_from_dataframe(
    validation_df,
    train_dir,
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size
)

#generating test data
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    test_dir,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

#create the convolutional neural network
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (2,2),strides=(2,2), activation='relu', input_shape=(image_width, image_height, image_channels)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(64, (2,2), strides=(2,2),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(128, (2,2),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(256, (2,2), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])



#callbacks - in case something goes wrong

#prevent overfitting by stopping learning if val_loss does not decrease after 10 epochs
earlystop = tf.keras.callbacks.EarlyStopping(patience=10)

#reduce learning rate if accuracy does not increase after 2 steps
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                               patience=2,
                                                               verbose=1,
                                                               factor=0.5,
                                                               min_lr=0.00001)
callbacks=[earlystop, learning_rate_reduction]
epochs = 2
model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=total_validation//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

#predict = model.predict_generator(test_generator, steps=np.ceil(total_test/batch_size))
test_df['guesses'] = 1#np.argmax(predict, axis=-1)
test_df['correct guesses'] = np.where(test_df['category'] == test_df['guesses'], 1, 0)
correct = sum(test_df['correct guesses'])
total = total_test
print(test_df.head())

print("The accuracy of the network on the test images is %s %%" %((100*correct)/total))



















