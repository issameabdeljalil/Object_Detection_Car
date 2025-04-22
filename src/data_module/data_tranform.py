import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from src.config import KITTI_CLASSES, KITTI_CLASS_TO_IDX

def parse_kitti_annotation(label_path):
    """
    Parse un fichier d'annotation KITTI
    
    Args:
        label_path: Chemin du fichier d'annotation
        
    Returns:
        Un dictionnaire contenant les boîtes, étiquettes et niveaux de difficulté
    """
    boxes = []
    labels = []
    difficulties = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            cls_name = parts[0]
            if cls_name == 'DontCare':
                continue

            # Niveau de difficulté (0 = facile, 1 = moyen, 2 = difficile)
            difficulty = int(float(parts[2]))

            # Coordonnées de la boîte [x1, y1, x2, y2]
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            boxes.append([x1, y1, x2, y2])
            labels.append(KITTI_CLASS_TO_IDX[cls_name])
            difficulties.append(difficulty)

    return {
        'boxes': torch.FloatTensor(boxes),
        'labels': torch.LongTensor(labels),
        'difficulties': torch.IntTensor(difficulties)
    }

class KittiDataset(Dataset):
    def __init__(self, root, split='training', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Chemins des dossiers d'images et d'annotations
        self.image_dir = os.path.join(root, split, 'image_2')
        self.label_dir = os.path.join(root, split, 'label_2')

        # Liste des fichiers d'images
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # On charge les annotations
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))
        target = parse_kitti_annotation(label_path)
        original_width, original_height = img.size

        if self.transform:
            img = self.transform(img)

        # Si l'image a été redimensionnée, on ajuste les boîtes
        if self.transform is not None and hasattr(self.transform, 'transforms'):
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    new_height, new_width = t.size
                    width_factor = new_width / original_width
                    height_factor = new_height / original_height

                    boxes = target['boxes']
                    boxes[:, 0] *= width_factor   # x1
                    boxes[:, 1] *= height_factor  # y1
                    boxes[:, 2] *= width_factor   # x2
                    boxes[:, 3] *= height_factor  # y2
                    target['boxes'] = boxes
                    break

        return img, target

def get_data_transforms(img_size=(375, 1242)):
    """
    Crée les transformations pour les données
    
    Args:
        img_size: Taille de redimensionnement des images
        
    Returns:
        Les transformations pour les données
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_data_loaders(data_dir, img_size=(375, 1242), batch_size=8):
    """
    Prépare les dataloaders pour l'entraînement et la validation
    
    Args:
        data_dir: Chemin du dossier contenant les données
        img_size: Taille de redimensionnement des images
        batch_size: Taille des batchs
        
    Returns:
        Dataloaders d'entraînement et de validation, ainsi que les datasets
    """
    transform = get_data_transforms(img_size)
    dataset = KittiDataset(root=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Nombre total d'échantillons: {len(dataset)}")
    print(f"Échantillons d'entraînement: {train_size}")
    print(f"Échantillons de validation: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: tuple(zip(*batch))  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    return dataset, train_dataset, val_dataset, train_loader, val_loader

def analyze_dataset(dataset):
    """
    Analyse la distribution des classes et la taille des boîtes dans le dataset
    
    Args:
        dataset: Dataset à analyser
        
    Returns:
        Un dictionnaire contenant les comptages de classes
    """
    class_counts = {cls_name: 0 for cls_name in KITTI_CLASSES}
    box_widths = []
    box_heights = []
    box_areas = []

    for _, target in tqdm(dataset, desc="Analyzing dataset"):
        labels = target['labels'].numpy()
        boxes = target['boxes'].numpy()

        for label, box in zip(labels, boxes):
            class_counts[KITTI_CLASSES[label]] += 1

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height

            box_widths.append(width)
            box_heights.append(height)
            box_areas.append(area)

    stats = {
        "class_counts": class_counts,
        "box_width_mean": np.mean(box_widths),
        "box_width_std": np.std(box_widths),
        "box_height_mean": np.mean(box_heights),
        "box_height_std": np.std(box_heights),
        "box_area_mean": np.mean(box_areas),
        "box_area_std": np.std(box_areas)
    }
    
    return stats