"""
WORKING IMAGE DOWNLOADER - Uses Bing Image Search
This WILL work - no complicated APIs needed!

Usage:
    pip install bing-image-downloader
    python working_download.py
"""

import os
import sys

print("="*80)
print("WORKING IMAGE DOWNLOADER")
print("="*80)
print("\nInstalling required package...")

# Install bing-image-downloader
import subprocess
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bing-image-downloader", "-q"])
    print("✅ Package installed!\n")
except:
    print("⚠️  Install manually: pip install bing-image-downloader")
    exit(1)

from bing_image_downloader import downloader

# Configuration
OUTPUT_DIR = "bing_dataset"
IMAGES_PER_CLASS = 200  # 200 images per class

# Search queries (specific to get good results)
SEARCHES = {
    "Television": "modern LED television TV screen",
    "Laptop": "laptop computer notebook",
    "Fridge": "refrigerator appliance fridge",
    "Washing_machine": "washing machine front load",
    "Air_Conditioner": "air conditioner AC unit split",
    "Mobile_Tablet": "smartphone mobile phone"
}

print("="*80)
print(f"Downloading {IMAGES_PER_CLASS} images per class")
print(f"Total: {IMAGES_PER_CLASS * len(SEARCHES)} images")
print("="*80 + "\n")

# Create base directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download each class
for class_name, search_query in SEARCHES.items():
    print(f"\n{'='*80}")
    print(f"📥 Downloading: {class_name}")
    print(f"   Search: '{search_query}'")
    print(f"{'='*80}\n")
    
    try:
        downloader.download(
            search_query,
            limit=IMAGES_PER_CLASS,
            output_dir=OUTPUT_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=False
        )
        
        # Rename the folder to our class name
        downloaded_folder = os.path.join(OUTPUT_DIR, search_query)
        target_folder = os.path.join(OUTPUT_DIR, class_name)
        
        if os.path.exists(downloaded_folder):
            if os.path.exists(target_folder):
                import shutil
                shutil.rmtree(target_folder)
            os.rename(downloaded_folder, target_folder)
            
            # Count images
            images = [f for f in os.listdir(target_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"✅ Downloaded {len(images)} images for {class_name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# Summary
print(f"\n{'='*80}")
print("📊 DOWNLOAD SUMMARY")
print(f"{'='*80}\n")

total = 0
for class_name in SEARCHES.keys():
    folder = os.path.join(OUTPUT_DIR, class_name)
    if os.path.exists(folder):
        images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        total += count
        print(f"✅ {class_name:20} : {count:4} images")
    else:
        print(f"❌ {class_name:20} : Failed")

print(f"\n{'='*80}")
print(f"Total: {total} images")
print(f"{'='*80}")

if total > 500:
    print(f"\n🎉 SUCCESS! Ready to train!")
    print(f"\n📁 Dataset: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\n🚀 NEXT STEP:")
    print(f"   python auto_train_no_checkpoint.py --source_dataset {OUTPUT_DIR}")
    print("="*80 + "\n")
else:
    print(f"\n⚠️  Only {total} images downloaded. You need at least 600 for decent training.")
    print("Try running the script again or use your existing dataset.")