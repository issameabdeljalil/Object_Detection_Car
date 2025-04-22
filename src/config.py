import os
import uuid
from datetime import datetime

# Création des dossiers nécessaires
os.makedirs("data", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)

# Définition des classes KITTI
KITTI_CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
KITTI_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(KITTI_CLASSES)}

experiment_id = str(uuid.uuid4())[:8]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


EXPERIMENT_CONFIG = {
    # Config du modèle
    "model_type": "fasterrcnn_resnet50_fpn",
    "experiment_name": f"fasterrcnn_{timestamp}_{experiment_id}",
    "techniques": ["focal_loss", "class_balanced_loss"],
    

    "epochs": 5,
    "batch_size": 8,
    "img_size": (375, 1242),  # Taille originale du dataset KITTI
    "optimizer": "SGD",
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "step_size": 3,
    "gamma": 0.1,
    
    # PATHS
    "data_dir": "./data",
    "output_dir": "./outputs",
    "checkpoint_dir": "./outputs/checkpoints",
    "metrics_dir": "./outputs/metrics",
    "visualizations_dir": "./outputs/visualizations",
    "save_period": 1,
    
    # Paramètres Wandb
    "use_wandb": False,
    "wandb_project": "kitti-object-detection",
    "wandb_api_key": None,  
}