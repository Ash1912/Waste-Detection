import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
from Model_Development.Transfer_Learning.config import Config

def get_data_generators():
    print("Setting up data generators...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        os.path.join(Config.BASE_DIR, 'train'),
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    print("Loading testing data...")
    test_gen = test_datagen.flow_from_directory(
        os.path.join(Config.BASE_DIR, 'test'),
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )
    return train_gen, test_gen

def build_model():
    print("Building model with VGG16 as the base...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3))
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model

def train_model():
    print("Initiating training process...")
    train_gen, test_gen = get_data_generators()
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compilation complete. Starting training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=Config.PATIENCE, restore_best_weights=True, verbose=1)
    history = model.fit(
        train_gen,
        steps_per_epoch=np.ceil(train_gen.samples / Config.BATCH_SIZE),
        epochs=Config.EPOCHS,
        validation_data=test_gen,
        validation_steps=np.ceil(test_gen.samples / Config.BATCH_SIZE),
        callbacks=[early_stopping]
    )

    # Saving the model
    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, 'transfer_learning_model.h5')
    model.save(model_save_path)
    print(f"Model saved successfully at {model_save_path}")

    print("Training completed. Evaluating model...")
    model.evaluate(test_gen)
    plot_training_history(history)

def plot_training_history(history):
    print("Plotting training and validation results...")
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'g*-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_loss, 'g*-', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    print("Results plotted successfully.")

if __name__ == '__main__':
    train_model()
