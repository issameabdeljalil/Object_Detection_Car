# config.py
# Configuration centrale du projet
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Kitti
import torchvision.transforms as transforms
import config
from src.data_module.data_transforms import get_train_transforms, get_val_test_transforms

# Chemins
DATA_ROOT = './data'
MODEL_SAVE_PATH = './models'
RESULTS_PATH = './results'

# Paramètres du dataset
IMG_SIZE = (375, 1242)
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Paramètres d'entraînement
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 30

# Paramètres du modèle
CONFIDENCE_THRESHOLD = 0.5

# Accès au GPU 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_datasets():
    # Download and prepare the training set
    train_dataset = Kitti(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=get_train_transforms()
    )
    
    # Download and prepare the test set
    test_dataset = Kitti(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=get_val_test_transforms()
    )
    
    # Create train/val split
    total_train_size = len(train_dataset)
    val_size = int((1 - config.TRAIN_VAL_SPLIT) * total_train_size)
    train_size = total_train_size - val_size
    
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    return train_subset, val_subset, test_dataset

def get_data_loaders():
    train_subset, val_subset, test_dataset = load_datasets()
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader

