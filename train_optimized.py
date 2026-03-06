"""
ReStoreAI Training with MobileNetV2
Lighter model that may generalize better

Usage:
    python train_mobilenet.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Larger batch for MobileNet
INITIAL_EPOCHS = 60
FINE_TUNE_EPOCHS = 40
NUM_CLASSES = 6
LEARNING_RATE = 0.001

DATASET_DIR = "dataset"

print("="*80)
print("RESTOREAI TRAINING - MobileNetV2")
print("="*80)
print(f"\nDataset: {DATASET_DIR}")
print(f"Model: MobileNetV2 (lighter, faster)")
print(f"Target: 66%+ accuracy")
print("="*80 + "\n")

if not os.path.exists(DATASET_DIR):
    print(f"❌ Error: {DATASET_DIR} not found!")
    sys.exit(1)

def create_model():
    """Create MobileNetV2 model"""
    print("\n🔨 Creating MobileNetV2 model...")
    
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Full width
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='MobileNetV2_DeviceClassifier')
    print("✅ Model created successfully")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model, base_model

def create_generators():
    """Create data generators with strong augmentation"""
    print("\n📊 Creating data generators...")
    
    # Strong augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.6, 1.4],
        channel_shift_range=30.0,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"\n✅ Generators created")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator

def train():
    """Main training function"""
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    
    # Create model and generators
    model, base_model = create_model()
    train_gen, val_gen = create_generators()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_mobilenet_weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n" + "="*80)
    print(f"PHASE 1: Training with frozen base ({INITIAL_EPOCHS} epochs)")
    print("="*80 + "\n")
    
    # Phase 1
    history = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    phase1_best = max(history.history['val_accuracy'])
    print(f"\n✅ Phase 1 complete - Best val_accuracy: {phase1_best:.4f} ({phase1_best*100:.2f}%)")
    
    # Phase 2 - Fine-tuning
    print("\n" + "="*80)
    print(f"PHASE 2: Fine-tuning ({FINE_TUNE_EPOCHS} epochs)")
    print("="*80 + "\n")
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    print(f"Unfroze {len([l for l in base_model.layers if l.trainable])} layers")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history.history['loss']),
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    phase2_best = max(history_fine.history['val_accuracy'])
    print(f"\n✅ Phase 2 complete - Best val_accuracy: {phase2_best:.4f} ({phase2_best*100:.2f}%)")
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING FINAL MODEL")
    print("="*80 + "\n")
    
    model.save_weights('models/mobilenet_final_weights.h5')
    print("✅ Final weights saved: models/mobilenet_final_weights.h5")
    
    try:
        model_json = model.to_json()
        with open('models/mobilenet_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        print("✅ Architecture saved: models/mobilenet_architecture.json")
    except:
        print("⚠️  Could not save architecture")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    results = model.evaluate(val_gen, verbose=0)
    best_accuracy = max(phase1_best, phase2_best)
    
    print(f"Final Validation Accuracy: {results[1]*100:.2f}%")
    print(f"Best Validation Accuracy: {best_accuracy*100:.2f}%")
    print("="*80)
    
    if best_accuracy >= 0.66:
        print("\n🎉 SUCCESS! Achieved 66%+ accuracy!")
    elif best_accuracy >= 0.60:
        print("\n✅ GOOD! Close to target (60%+)")
    else:
        print(f"\n⚠️  Got {best_accuracy*100:.1f}% - below target")
    
    print("\n📁 Model files:")
    print("   ✅ models/best_mobilenet_weights.h5 (best)")
    print("   ✅ models/mobilenet_final_weights.h5 (final)")
    print("\n📝 Next: python app.py")
    print("="*80 + "\n")
    
    return model

if __name__ == "__main__":
    try:
        print("\n⏰ Starting at:", tf.timestamp())
        model = train()
        print("\n✅ Training complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()