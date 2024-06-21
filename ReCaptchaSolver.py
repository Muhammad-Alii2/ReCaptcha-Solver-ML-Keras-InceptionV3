import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import applications, optimizers
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Dataset downloaded from: https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images

# Parameters
image_height = 150
image_width = 150
batch_size = 64
num_classes = 13  # Number of classes based on the folder names

# Create an ImageDataGenerator instance for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of the data for validation
)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    'D:\\ReCaptchaML\\images',
    target_size=(image_height, image_width),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

# Load and preprocess validation data
validation_generator = train_datagen.flow_from_directory(
    'D:\\ReCaptchaML\\images',
    target_size=(image_height, image_width),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Define the model
ROWS = 150
COLS = 150
input_shape = (ROWS, COLS, 3)

base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, 3))

for i in base_model.layers[0:150]:
    i.trainable = False
for i in base_model.layers[150::]:
    i.trainable = True

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(13, activation='softmax'))


# Compile the model
model = add_model

model = load_model('bestModel.keras')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])
model.summary()
print(len(base_model.layers))

# Train the model using the generators
callback = ModelCheckpoint(
    filepath='bestModel.keras',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose =1)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    callbacks=callback,
    epochs=10)

# Evaluate the model
model.evaluate(validation_generator)
