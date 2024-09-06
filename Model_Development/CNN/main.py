import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from Model_Development.CNN.config import Config

logging.basicConfig(level=logging.INFO)

def get_data_generators():

    # Data Augmentation setup
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

    test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation data

    print("Loading training data...")
    # Generate data from directories
    train_data_gen = train_datagen.flow_from_directory(
        directory=os.path.join(Config.BASE_DIR, 'train'),
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    print("Loading testing data...")
    test_data_gen = test_datagen.flow_from_directory(
        directory=os.path.join(Config.BASE_DIR, 'test'),
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    print("Training batches: ", len(train_data_gen))
    print("Validation batches: ", len(test_data_gen))

    return train_data_gen, test_data_gen


def cnn_model():
    print("Initializing data augmentation setup...")
    train_data_gen, test_data_gen = get_data_generators()

    print("Constructing the CNN model...")
    # Enhanced CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Early Stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=Config.PATIENCE, restore_best_weights=True, verbose=1)

    print("Starting model training...")
    # Model Training
    history = model.fit(
        train_data_gen,
        steps_per_epoch=np.ceil(train_data_gen.samples / Config.BATCH_SIZE),
        epochs=Config.EPOCHS,
        validation_data=test_data_gen,
        validation_steps=np.ceil(test_data_gen.samples / Config.BATCH_SIZE),
        callbacks=[early_stopping]
    )

    print("Model training completed.")

    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, 'cnn_model.h5')
    model.save(model_save_path)
    logging.info(f"Model saved successfully at {model_save_path}")

    evaluate_model(model, test_data_gen)
    plot_training_history(history)

def evaluate_model(model, test_data_gen):
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_data_gen)
    logging.info(f'Test accuracy: {test_accuracy * 100:.2f}%, Test loss: {test_loss:.4f}')

def plot_training_history(history):
    print("Plotting training history...")
    train_acc = history.history['accuracy'] # store training accuracy in history
    val_acc = history.history['val_accuracy'] # store validation accuracy in history
    train_loss = history.history['loss'] # store training loss in history
    val_loss = history.history['val_loss'] # store validation loss in history

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'g*-', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_loss, 'g*-', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    print("Training history plotted successfully.")

if __name__ == '__main__':
    cnn_model()
