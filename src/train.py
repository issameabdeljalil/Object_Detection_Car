import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from src.config import KITTI_CLASSES, EXPERIMENT_CONFIG
from src.download_dataset import download_kitti_dataset
from src.data_module.data_transform import prepare_data_loaders, analyze_dataset
from src.models import get_detection_model, save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.evaluate import evaluate_model

def init_wandb(config):
    """
    Initialise Weights & Biases pour le suivi des expériences
    
    Args:
        config: Configuration de l'expérience
        
    Returns:
        L'ID de la run W&B
    """
    if config["use_wandb"]:
        print("Initialisation de Weights & Biases...")
        if config["wandb_api_key"]:
            os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
            
        wandb_run = wandb.init(
            project=config["wandb_project"],
            name=config["experiment_name"],
            config=config,
            resume="allow"
        )
        print(f"Run W&B initialisé: {config['experiment_name']}")
        return wandb_run.id
    return None

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Entraîne le modèle pendant une époque
    
    Args:
        model: Le modèle à entraîner
        optimizer: L'optimiseur à utiliser
        data_loader: Le dataloader d'entraînement
        device: Le dispositif sur lequel entraîner le modèle
        epoch: Le numéro de l'époque
        
    Returns:
        La perte moyenne pour cette époque
    """
    model.train()

    running_loss = 0.0
    epoch_loss = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")

    for images, targets in progress_bar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'difficulties'} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Backward pass et optimisation
        losses.backward()
        optimizer.step()

        running_loss += loss_value
        epoch_loss += loss_value
        progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1)})

    return epoch_loss / len(data_loader)

def train_model(config=None):
    """
    Fonction principale d'entraînement du modèle
    
    Args:
        config: Configuration de l'expérience (utilise la configuration par défaut si non spécifiée)
        
    Returns:
        Le modèle entraîné, l'historique des pertes et des métriques
    """
    if config is None:
        config = EXPERIMENT_CONFIG
    data_dir = download_kitti_dataset(config["data_dir"])
    dataset, train_dataset, val_dataset, train_loader, val_loader = prepare_data_loaders(
        data_dir, 
        img_size=config["img_size"], 
        batch_size=config["batch_size"]
    )
    
    if config.get("analyze_dataset", True):
        stats = analyze_dataset(dataset)
        print("Distribution des classes:")
        for cls, count in stats["class_counts"].items():
            print(f"{cls}: {count}")
    
    # GPU ou CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositif utilisé: {device}")
    
    # Initialiser W&B 
    if config["use_wandb"]:
        wandb_run_id = init_wandb(config)
    
    model = get_detection_model(num_classes=len(KITTI_CLASSES))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    if config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(params, lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Optimiseur non pris en charge: {config['optimizer']}")
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    checkpoint_path, start_epoch = find_latest_checkpoint(config)
    
    if checkpoint_path:
        model, optimizer, start_epoch, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"Checkpoint chargé, reprise de l'entraînement à l'époque {start_epoch+1}")
    else:
        start_epoch = 0
        print("Démarrage d'un nouvel entraînement")
    
    # outputs
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    num_epochs = config["epochs"]
    train_losses = []
    val_metrics_history = []
    
    print(f"Début de l'entraînement pour {num_epochs} époques")
    
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_losses.append(train_loss)
        lr_scheduler.step()
        
        # Évaluer perfs
        metrics = evaluate_model(model, val_loader, device)
        val_metrics_history.append(metrics)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | mAP@0.5: {metrics['mAP_50']:.4f} | mAP@0.75: {metrics['mAP_75']:.4f}")
        
        # Logging Wandb
        if config["use_wandb"]:
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mAP_50": metrics["mAP_50"],
                "mAP_75": metrics["mAP_75"],
                "mAP": metrics["mAP"]
            }
            for cls_name, cls_metrics in metrics["class_metrics"].items():
                if cls_name != "DontCare":
                    wandb_log[f"AP_{cls_name}"] = cls_metrics["AP"]
                    wandb_log[f"precision_{cls_name}"] = cls_metrics["precision"]
                    wandb_log[f"recall_{cls_name}"] = cls_metrics["recall"]
                    wandb_log[f"f1_{cls_name}"] = cls_metrics["f1"]
            
            wandb.log(wandb_log)
        
        # Sauvegarder le modèle périodiquement
        if (epoch + 1) % config["save_period"] == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, metrics, config)
    plot_training_metrics(train_losses, val_metrics_history, config["visualizations_dir"])
    final_model_path = os.path.join(config["output_dir"], f"{config['experiment_name']}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modèle final sauvegardé à {final_model_path}")
    
    return model, train_losses, val_metrics_history

def plot_training_metrics(train_losses, val_metrics_history, output_dir):
    """
    Trace les métriques d'entraînement
    
    Args:
        train_losses: Liste des pertes d'entraînement
        val_metrics_history: Historique des métriques de validation
        output_dir: Dossier de sortie pour les visualisations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Perte d\'entraînement')
    plt.xlabel('Époques')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot([m['mAP_50'] for m in val_metrics_history], label='mAP@0.5')
    plt.plot([m['mAP_75'] for m in val_metrics_history], label='mAP@0.75')
    plt.title('Métriques de validation')
    plt.xlabel('Époques')
    plt.ylabel('mAP')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

if __name__ == "__main__":
    train_model()