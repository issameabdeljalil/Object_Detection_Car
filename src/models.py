import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def get_detection_model(num_classes):
    """
    Crée un modèle Faster R-CNN pour la détection d'objets
    
    Args:
        num_classes: Nombre de classes à détecter
        
    Returns:
        Le modèle Faster R-CNN
    """
    print("Initialisation du modèle Faster R-CNN ResNet-50...")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def save_checkpoint(model, optimizer, epoch, loss, metrics, config, filename=None):
    """
    Sauvegarde un point de contrôle du modèle
    
    Args:
        model: Le modèle à sauvegarder
        optimizer: L'optimiseur à sauvegarder
        epoch: Numéro de l'époque
        loss: Valeur de perte
        metrics: Métriques d'évaluation
        config: Configuration de l'expérience
        filename: Nom du fichier de sauvegarde (optionnel)
        
    Returns:
        Le chemin du fichier de sauvegarde
    """
    import os
    
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"{config['experiment_name']}_epoch_{epoch}.pth"
    
    checkpoint_file = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved to {checkpoint_file}")
    return checkpoint_file

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Charge un point de contrôle du modèle
    
    Args:
        checkpoint_path: Chemin du fichier de point de contrôle
        model: Le modèle dans lequel charger les poids
        optimizer: L'optimiseur à mettre à jour (optionnel)
        device: Dispositif sur lequel charger le modèle
        
    Returns:
        Le modèle mis à jour, l'optimiseur, l'époque, la perte, les métriques et la configuration
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    metrics = checkpoint.get('metrics', {})
    config = checkpoint.get('config', {})
    
    return model, optimizer, epoch, loss, metrics, config

def find_latest_checkpoint(config):
    """
    Recherche le dernier point de contrôle sauvegardé
    
    Args:
        config: Configuration de l'expérience
        
    Returns:
        Le chemin du dernier point de contrôle et son époque
    """
    import os
    
    checkpoint_dir = config["checkpoint_dir"]
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".pth") and config["experiment_name"] in file:
                try:
                    epoch = int(file.split("_epoch_")[-1].replace(".pth", ""))
                    checkpoints.append((file, epoch, os.path.join(checkpoint_dir, file)))
                except ValueError:
                    pass
    
    if not checkpoints:
        return None, 0
    
    latest_file, latest_epoch, latest_path = max(checkpoints, key=lambda x: x[1])
    print(f"Dernier point de contrôle trouvé: {latest_file} (Epoch {latest_epoch})")
    return latest_path, latest_epoch