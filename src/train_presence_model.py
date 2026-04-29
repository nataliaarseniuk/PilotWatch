import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
DATASET_DIR = '../pilot_presence_dataset'
MODEL_OUTPUT = 'pilot_presence_model.h5'

def check_dataset():
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"\nERROR: {DATASET_DIR}/ not found!")
        return False
    
    present_count = len([f for f in os.listdir(os.path.join(DATASET_DIR, 'pilot_present')) if f.endswith('.jpg')])
    absent_count = len([f for f in os.listdir(os.path.join(DATASET_DIR, 'no_pilot')) if f.endswith('.jpg')])
    
    print(f"\nPilot present: {present_count} images")
    print(f"No pilot: {absent_count} images")
    print(f"Total: {present_count + absent_count} images")

    return True

def create_model():
    """
    This function builds a Convolutional Neural Network (CNN) for binary classification to detect pilot presence.
    Architecture:

    3 Convolutional Blocks - Extract visual features from images at different levels of complexity

    Block 1: 32 filters detect basic patterns (edges, corners)
    Block 2: 64 filters detect intermediate features (shapes, textures)
    Block 3: 128 filters detect complex features (facial structures)
    Each block includes: Conv2D → BatchNormalization → MaxPooling (downsampling) → Dropout (prevent overfitting)

    2 Dense Layers - Combine features to make classification decision

    128 neurons and 64 neurons with dropout for final feature processing

    Output Layer - Single neuron with sigmoid activation

    Outputs value between 0-1 representing confidence (0 = no pilot, 1 = pilot present)

    Compilation:

    Uses Adam optimizer with 0.001 learning rate
    Binary crossentropy loss function (for yes/no classification)
    Tracks accuracy metric during training

    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    print("\n" + "="*60)
    print("TRAINING PILOT PRESENCE MODEL")
    print("="*60)
    
    if not check_dataset():
        return
    
    # Simple data generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    print("\nLoading training data...")
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    print("Loading validation data...")
    val_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    print("\nClass mapping:", train_gen.class_indices)
    
    # Build model
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_OUTPUT,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Training for up to {EPOCHS} epochs...")
    print("(Early stopping if validation stops improving)\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot results
    print("\nGenerating training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('presence_training_history.png', dpi=150)
    print("Saved: presence_training_history.png")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTraining Accuracy:   {final_train_acc*100:.2f}%")
    print(f"Validation Accuracy: {final_val_acc*100:.2f}%")

    print(f"\nModel saved to: {MODEL_OUTPUT}")

if __name__ == "__main__":
    train()
