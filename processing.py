import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

source_dirs = {
    "train_a": "dataset/train_a",
    "train_b": "dataset/train_b"
}
target_base = "processed"

# Define transform: resize + to tensor + normalize
transform = transforms.Compose([
    transforms.Resize((448,448)),                  # Resize to 256x256
    transforms.ToTensor(),                          # Convert to tensor
    ])

# Function to process images in a folder
def preprocess_folder(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename in tqdm(os.listdir(source_dir), desc=f"Processing {source_dir}"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(source_dir, filename)
            image = Image.open(path).convert("RGB")
            image = transform(image)
            # Convert back to PIL image for saving
            image = transforms.ToPILImage()(image)
            image.save(os.path.join(target_dir, filename))

# Run preprocessing for both trainA and trainB
for key, src_dir in source_dirs.items():
    tgt_dir = os.path.join(target_base, key)
    preprocess_folder(src_dir, tgt_dir)

print("✅ Preprocessing complete! Images saved to 'processed/trainA' and 'processed/trainB'")
