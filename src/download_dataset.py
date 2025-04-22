import os
import torch
from torchvision.datasets import Kitti
from tqdm import tqdm

def download_kitti_dataset(data_dir="./data"):
    """
    Télécharge le dataset KITTI
    
    Args:
        data_dir: Chemin du dossier où télécharger le dataset
    """
    print("Téléchargement du dataset KITTI...")
    
    # Vérifier si le dossier Kitti existe déjà
    if not os.path.exists(os.path.join(data_dir, "Kitti", "raw")):
        os.makedirs(data_dir, exist_ok=True)
        
        print("Téléchargement du dataset d'entraînement...")
        train_dataset = Kitti(
            root=data_dir,
            train=True,
            download=True,
            transform=None  # Pas de transformation ici pour garder le format original
        )
        
        print("Téléchargement du dataset de test...")
        test_dataset = Kitti(
            root=data_dir,
            train=False,
            download=True,
            transform=None
        )
        
        print("Téléchargement terminé !")
    else:
        print("Le dataset KITTI est déjà téléchargé.")
    
    # Vérifier le nombre d'images
    train_dir = os.path.join(data_dir, "Kitti", "raw", "training", "image_2")
    if os.path.exists(train_dir):
        num_images = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
        print(f"Nombre d'images d'entraînement: {num_images}")
    else:
        print("Attention: Le dossier d'images d'entraînement n'a pas été trouvé.")
    
    return os.path.join(data_dir, "Kitti", "raw")

if __name__ == "__main__":
    kitti_path = download_kitti_dataset()
    print(f"Dataset KITTI téléchargé à {kitti_path}")