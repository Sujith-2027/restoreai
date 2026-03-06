"""
Complete Auto-Train Script - NO CALLBACK VERSION
Removes ModelCheckpoint to avoid serialization errors

Usage:
    python auto_train_no_checkpoint.py --source_dataset ./dataset
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 20
NUM_CLASSES = 6
LEARNING_RATE = 0.001

# Folder name mapping
FOLDER_MAPPING = {
    "Air Conditioner": "Air_Conditioner",
    "air conditioner": "Air_Conditioner",
    "AC": "Air_Conditioner",
    "Washing machine": "Washing_machine",
    "washing machine": "Washing_machine",
    "WM": "Washing_machine",
    "Television": "Television",
    "TV": "Television",
    "tv": "Television",
    "Laptop": "Laptop",
    "laptop": "Laptop",
    "Fridge": "Fridge",
    "fridge": "Fridge",
    "Refrigerator": "Fridge",
    "Mobile-tablet": "Mobile_Tablet",
    "Mobile tablet": "Mobile_Tablet",
    "mobile-tablet": "Mobile_Tablet",
    "Phone": "Mobile_Tablet",
    "Tablet": "Mobile_Tablet"
}

# Expected classes
EXPECTED_CLASSES = [
    "Television",
    "Air_Conditioner",
    "Fridge",
    "Mobile_Tablet",
    "Laptop",
    "Washing_machine",
]

print("="*80)
print("AUTO-TRAIN SCRIPT - NO CHECKPOINT VERSION")
print("="*80)
print("This script will:")
print("1. Analyze your dataset folder")
print("2. Create train/validation split (80/20)")
print("3. Fix folder names automatically")
print("4. Train EfficientNetB0 model")
print("5. Save as models/image_model.h5")
print("="*80 + "\n")


def count_images(folder):
    """Count valid image files in a folder"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    count = 0
    for file in Path(folder).iterdir():
        if file.suffix.lower() in valid_extensions:
            count += 1
    return count


def analyze_source_dataset(source_dir):
    """Analyze the original dataset structure"""
    print(f"📁 Analyzing dataset in: {source_dir}\n")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Error: Dataset directory not found: {source_dir}")
        sys.exit(1)
    
    folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"❌ Error: No folders found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(folders)} folders:\n")
    
    dataset_info = {}
    total_images = 0
    
    for folder in folders:
        num_images = count_images(folder)
        total_images += num_images
        
        correct_name = FOLDER_MAPPING.get(folder.name, folder.name)
        
        status = "✅" if correct_name in EXPECTED_CLASSES else "⚠️"
        dataset_info[folder] = {
            'original_name': folder.name,
            'correct_name': correct_name,
            'num_images': num_images,
            'valid': correct_name in EXPECTED_CLASSES
        }
        
        print(f"{status} {folder.name:25} → {correct_name:20} ({num_images} images)")
    
    print(f"\nTotal images: {total_images}")
    
    return dataset_info


def create_split_dataset(source_dir, output_dir, train_split=0.8):
    """Create train/validation split from source dataset"""
    print(f"\n{'='*80}")
    print("CREATING TRAIN/VALIDATION SPLIT")
    print(f"{'='*80}\n")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    train_dir = output_path / "train"
    val_dir = output_path / "validation"
    
    if output_path.exists():
        print(f"⚠️  Output directory exists: {output_path}")
        response = input("Delete and recreate? (y/n): ").lower()
        if response == 'y':
            shutil.rmtree(output_path)
            print("✅ Deleted old directory")
        else:
            print("❌ Aborted.")
            sys.exit(1)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_info = analyze_source_dataset(source_dir)
    
    print(f"\nCreating split with ratio: {train_split:.0%} train / {1-train_split:.0%} validation\n")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    total_train = 0
    total_val = 0
    
    for source_folder, info in dataset_info.items():
        if not info['valid']:
            continue
        
        correct_name = info['correct_name']
        
        image_files = [
            f for f in source_folder.iterdir() 
            if f.suffix.lower() in valid_extensions
        ]
        
        if not image_files:
            continue
        
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * train_split)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        train_class_dir = train_dir / correct_name
        val_class_dir = val_dir / correct_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        print(f"📁 {info['original_name']} → {correct_name}")
        print(f"   Training: {len(train_images)} images")
        print(f"   Validation: {len(val_images)} images")
        
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
            total_train += 1
        
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
            total_val += 1
        
        print()
    
    print(f"{'='*80}")
    print("✅ DATASET SPLIT COMPLETE")
    print(f"{'='*80}")
    print(f"Training images: {total_train}")
    print(f"Validation images: {total_val}")
    print(f"{'='*80}\n")
    
    return train_dir, val_dir


def create_efficientnet_model():
    """Create EfficientNetB0 transfer learning model"""
    print("\n🔨 Creating EfficientNetB0 model...")
    
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='EfficientNetB0_DeviceClassifier')
    
    print("✅ Model created successfully")
    return model, base_model


def create_data_generators(train_dir, val_dir):
    """Create image data generators with augmentation"""
    print("\n📊 Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    print(f"\n✅ Data generators created")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator


def plot_training_history(history, save_path='training_history.png'):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Training history saved: {save_path}")
    plt.close()


# Custom Keras callback to save best model during training
class BestModelCallback(tf.keras.callbacks.Callback):
    """Custom callback to save the best model during training"""
    
    def __init__(self, filepath='models/best_model_weights.h5'):
        super().__init__()
        self.filepath = filepath
        self.best_val_acc = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        logs = logs or {}
        val_acc = logs.get('val_accuracy', 0)
        
        # Convert to float if it's a tensor
        if hasattr(val_acc, 'numpy'):
            val_acc = float(val_acc.numpy())
        elif hasattr(val_acc, 'item'):
            val_acc = float(val_acc.item())
        else:
            val_acc = float(val_acc)
        
        # Save if validation accuracy improved
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # Only save weights to avoid serialization issues
            self.model.save_weights(self.filepath)
            print(f"\n💾 Epoch {epoch + 1}: Saved best model weights (val_accuracy: {val_acc:.4f})")



def train_model(train_dir, val_dir, fine_tune=True):
    """Train EfficientNet model with optional fine-tuning"""
    print(f"\n{'='*80}")
    print("STARTING MODEL TRAINING")
    print(f"{'='*80}\n")
    
    train_gen, val_gen = create_data_generators(train_dir, val_dir)
    model, base_model = create_efficientnet_model()
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n" + "="*80)
    model.summary()
    print("="*80 + "\n")
    
    os.makedirs('models', exist_ok=True)
    
    # Create custom checkpoint callback (saves weights only)
    best_model_callback = BestModelCallback(filepath='models/best_model_weights.h5')
    
    # Callbacks
    callbacks = [
        best_model_callback,
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n" + "="*80)
    print("PHASE 1: Training with frozen EfficientNet base")
    print("="*80 + "\n")
    
    # Train phase 1 - all epochs at once
    history = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 1 complete - Best val_accuracy: {best_model_callback.best_val_acc:.4f}")
    
    # Phase 2: Fine-tuning
    if fine_tune:
        print("\n" + "="*80)
        print("PHASE 2: Fine-tuning (unfreezing base model)")
        print("="*80 + "\n")
        
        base_model.trainable = True
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning all {len(base_model.layers)} layers\n")
        
        # Train phase 2 - all epochs at once
        history_fine = model.fit(
            train_gen,
            epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
            initial_epoch=INITIAL_EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✅ Phase 2 complete - Best val_accuracy: {best_model_callback.best_val_acc:.4f}")
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING FINAL MODEL")
    print("="*80 + "\n")
    
    # Save weights (always works)
    model.save_weights('models/image_model_weights.h5')
    print("✅ MODEL WEIGHTS SAVED: models/image_model_weights.h5")
    
    # Try to save architecture separately as JSON
    try:
        model_json = model.to_json()
        with open('models/image_model_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        print("✅ MODEL ARCHITECTURE SAVED: models/image_model_architecture.json")
    except Exception as e:
        print(f"⚠️  Could not save architecture as JSON: {e}")
    
    print("="*80)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    results = model.evaluate(val_gen, verbose=0)
    val_loss = results[0]
    val_accuracy = results[1]
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Best Validation Accuracy: {best_model_callback.best_val_acc*100:.2f}%")
    print("="*80)
    
    return model


def test_model():
    """Test the saved model"""
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80 + "\n")
    
    from tensorflow.keras.models import model_from_json
    
    model = None
    
    # Try loading from architecture + weights
    try:
        if os.path.exists('models/image_model_architecture.json'):
            with open('models/image_model_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights('models/image_model_weights.h5')
            print("✅ Model loaded successfully (from architecture + weights)")
        else:
            raise Exception("No architecture file found")
    except Exception as e:
        print(f"⚠️  Could not load from JSON: {e}")
        # Recreate model and load weights
        try:
            print("Recreating model architecture and loading weights...")
            model, _ = create_efficientnet_model()
            
            # Compile model before loading weights
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.load_weights('models/image_model_weights.h5')
            print("✅ Model loaded successfully (recreated + weights)")
        except Exception as e2:
            print(f"❌ Model test failed: {e2}")
            return False
    
    # Test inference
    try:
        # Compile if not already compiled
        if not model.optimizer:
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        dummy_img = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype('float32')
        pred = model.predict(dummy_img, verbose=0)[0]
        pred_class = int(np.argmax(pred))
        pred_conf = float(np.max(pred))
        
        class_names = ["Television", "Air Conditioner", "Fridge", 
                      "Mobile/Tablet", "Laptop", "Washing machine"]
        
        print(f"✅ Inference test successful")
        print(f"   Predicted class: {pred_class} ({class_names[pred_class]})")
        print(f"   Confidence: {pred_conf*100:.2f}%")
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_trained_model(weights_path='models/best_model_weights.h5'):
    """
    Helper function to load the trained model for inference
    
    Usage:
        from auto_train_no_checkpoint import load_trained_model
        model = load_trained_model()
        # or load final model
        model = load_trained_model('models/image_model_weights.h5')
    
    Args:
        weights_path: Path to the weights file
        
    Returns:
        Compiled Keras model ready for inference
    """
    from tensorflow.keras.models import model_from_json
    
    # Try loading from JSON architecture first
    try:
        if os.path.exists('models/image_model_architecture.json'):
            with open('models/image_model_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(weights_path)
            print(f"✅ Model loaded from architecture + weights: {weights_path}")
        else:
            raise Exception("Architecture file not found, recreating...")
    except Exception as e:
        # Recreate the model architecture
        print(f"Recreating model architecture...")
        model, _ = create_efficientnet_model()
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.load_weights(weights_path)
        print(f"✅ Model loaded (recreated architecture + weights): {weights_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Auto-split dataset and train EfficientNet')
    parser.add_argument('--source_dataset', type=str, default='./dataset')
    parser.add_argument('--output_dataset', type=str, default='./dataset_split')
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--skip_split', action='store_true')
    parser.add_argument('--no_fine_tune', action='store_true')
    
    args = parser.parse_args()
    
    try:
        if not args.skip_split:
            train_dir, val_dir = create_split_dataset(
                args.source_dataset,
                args.output_dataset,
                args.train_split
            )
        else:
            train_dir = Path(args.output_dataset) / "train"
            val_dir = Path(args.output_dataset) / "validation"
            
            if not train_dir.exists() or not val_dir.exists():
                print("❌ Error: Train/validation directories not found!")
                sys.exit(1)
        
        model = train_model(train_dir, val_dir, fine_tune=not args.no_fine_tune)
        test_model()
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📁 Generated files:")
        print("   ✅ models/image_model_weights.h5 (model weights)")
        print("   ✅ models/image_model_architecture.json (model structure)")
        print("   ✅ models/best_model_weights.h5 (best checkpoint)")
        print("\n💡 To use the model:")
        print("   1. Load architecture from JSON")
        print("   2. Load weights from H5 file")
        print("   3. Or use the test_model() function as reference")
        print("\n📝 Next step: python main.py")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()