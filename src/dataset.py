import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ORCADataset(Dataset):
    def __init__(self, root_dir, subset='training', image_size=(512, 512)):
        self.images_dir = os.path.join(root_dir, subset, 'tumor')
        self.masks_dir = os.path.join(root_dir, subset, 'lesion_annotations')

        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

        self.images = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(valid_exts)
        ])

        self.masks = sorted([
            f for f in os.listdir(self.masks_dir)
            if f.lower().endswith(valid_exts)
        ])

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # Normalização compatível com ImageNet (MobileNet V3 pré-treinado)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            # Evita criar valores intermediários na máscara ao redimensionar
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        # Binariza a máscara (0/1) antes de converter para inteiro de classe
        mask = (mask > 0.5).squeeze(0).long()

        return image, mask
