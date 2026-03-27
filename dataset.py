from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_A = os.path.join(root_dir, 'train_a')  # match your folder
        self.root_B = os.path.join(root_dir, 'train_b')
        self.files_A = sorted(os.listdir(self.root_A))
        self.files_B = sorted(os.listdir(self.root_B))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = Image.open(os.path.join(self.root_A, self.files_A[index % len(self.files_A)])).convert("RGB")
        img_B = Image.open(os.path.join(self.root_B, self.files_B[index % len(self.files_B)])).convert("RGB")

        return {
            'A': self.transform(img_A),
            'B': self.transform(img_B)
        }
