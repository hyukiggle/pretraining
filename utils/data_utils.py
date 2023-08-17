import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Define your data paths
imagenet_root = "/home/hyukiggle/Documents/data/ImageNet/train/"
output_root = "/path/to/output/inpainting/dataset/"

# Define the mask size and percentage of pixels to mask
mask_size = (64, 64)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])

# Define a custom dataset for inpainting
class InpaintingDataset(Dataset):
    def __init__(self, root_dir, mask_size = (64, 64), transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.mask_size = mask_size
        
        for root, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_list.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert("RGB")
    
        if self.transform:
            image = self.transform(image)
        
        masked_image = self.apply_mask(image)

        return masked_image, image

    def apply_mask(self, image):
        masked_image = image.clone()
        h_offset = random.randint(image.shape[1]//3, image.shape[1]//3 * 2)
        w_offset = random.randint(image.shape[1]//3, image.shape[1]//3 * 2)
        masked_image[:, h_offset:h_offset+mask_size[0], w_offset:w_offset+mask_size[1]] = 0
        return masked_image


# if __name__ == "__main__":
#     # Create the dataset
#     dataset = InpaintingDataset('/home/hyukiggle/Documents/data/ImageNet/train', transform=transform)

#     # Create a DataLoader
#     batch_size = 32
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     masked_image, image = next(iter(dataloader))
#     print(masked_image.shape)
#     temp = masked_image[3].permute(1, 2, 0).numpy()
#     plt.imshow(temp)
#     plt.show()
#     plt.imshow(image[3].permute(1, 2, 0).numpy())
#     plt.show()