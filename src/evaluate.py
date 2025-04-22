import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou

from src.config import KITTI_CLASSES, EXPERIMENT_CONFIG

def evaluate_model(model, data_loader, device, save_metrics=False, output_dir=None, config=None):
    """
    Évalue le modèle sur un ensemble de données
    
    Args:
        model: Modèle à évaluer
        data_loader: DataLoader pour l'évaluation
        device: Dispositif sur lequel évaluer le modèle
        save_metrics: Indique s'il faut sauvegarder les métriques
        output_dir: Dossier où sauvegarder les métriques
        config: Configuration de l'expérience
        
    Returns:
        Un dictionnaire contenant les métriques d'évaluation
    """
    if config is None:
        config = EXPERIMENT_CONFIG
        
    if output_dir is None:
        output_dir = config["metrics_dir"]
        
    model.eval()

    iou_thresholds = [0.5, 0.75]
    metrics = {
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP": 0.0,
        "detections": 0,
        "ground_truths": 0
    }

    class_metrics = {cls: {"TP": 0, "FP": 0, "FN": 0, "AP": 0.0} for cls in KITTI_CLASSES[:-1]}  # Ignorer 'DontCare'

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Évaluation"):
            images = list(img.to(device) for img in images)
            targets_cpu = [{k: v for k, v in t.items() if k != 'difficulties'} for t in targets]
            outputs = model(images)

            # Calculer les métriques pour chaque image
            for img_idx, (output, target) in enumerate(zip(outputs, targets_cpu)):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()
                gt_boxes = target['boxes']
                gt_labels = target['labels']

                metrics["detections"] += len(pred_boxes)
                metrics["ground_truths"] += len(gt_boxes)

                # Calculer l'AP pour chaque classe présente dans cette image
                for cls_idx in range(len(KITTI_CLASSES) - 1):  # Ignorer 'DontCare'
                    pred_class_mask = pred_labels == cls_idx
                    gt_class_mask = gt_labels == cls_idx

                    pred_class_boxes = pred_boxes[pred_class_mask]
                    pred_class_scores = pred_scores[pred_class_mask]
                    gt_class_boxes = gt_boxes[gt_class_mask]
                    
                    # Compter les FN pour cette classe
                    class_metrics[KITTI_CLASSES[cls_idx]]["FN"] += len(gt_class_boxes)

                    if len(gt_class_boxes) == 0:
                        continue  
                    # Calculer l'AP pour cette classe et cette image
                    for iou_idx, iou_threshold in enumerate(iou_thresholds):
                        if len(pred_class_boxes) == 0:
                            continue  

                        # Calculer l'IoU entre les prédictions et ground truth
                        ious = box_iou(pred_class_boxes, gt_class_boxes)

                        # Pour chaque prédiction, prendre la vérité terrain avec l'IoU maximum
                        max_ious, _ = ious.max(dim=1)

                        # Seuil de score pour considérer une prédiction valide
                        score_threshold = 0.5
                        valid_preds = pred_class_scores >= score_threshold
                        sorted_indices = torch.argsort(pred_class_scores, descending=True)
                        max_ious = max_ious[sorted_indices]
                        valid_preds = valid_preds[sorted_indices]

                        # Calculer les vrais positifs et faux positifs
                        tp = ((max_ious >= iou_threshold) & valid_preds).float()
                        fp = ((max_ious < iou_threshold) & valid_preds).float()
                        class_name = KITTI_CLASSES[cls_idx]
                        class_metrics[class_name]["TP"] += tp.sum().item()
                        class_metrics[class_name]["FP"] += fp.sum().item()
                        class_metrics[class_name]["FN"] -= tp.sum().item()  # Réduire le nombre de FN par le nombre de TP

                        # Précisions et rappels cumulatifs
                        tp_cumsum = torch.cumsum(tp, dim=0)
                        fp_cumsum = torch.cumsum(fp, dim=0)

                        # Calculer les précisions et rappels pour chaque seuil
                        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
                        recalls = tp_cumsum / (len(gt_class_boxes) + 1e-10)

                        # Ajouter le point (0, 1) pour commencer la courbe PR
                        precisions = torch.cat([torch.tensor([1]), precisions])
                        recalls = torch.cat([torch.tensor([0]), recalls])

                        # Calculer l'aire sous la courbe PR (AP)
                        ap = torch.trapz(precisions, recalls)
                        
                        # Mettre à jour l'AP pour cette classe
                        class_metrics[class_name]["AP"] += ap.item() / len(data_loader)

                        # Ajouter l'AP à la métrique correspondante
                        if iou_idx == 0:  # IoU = 0.5
                            metrics["mAP_50"] += ap.item() / (len(data_loader) * (len(KITTI_CLASSES) - 1))
                        elif iou_idx == 1:  # IoU = 0.75
                            metrics["mAP_75"] += ap.item() / (len(data_loader) * (len(KITTI_CLASSES) - 1))

    # Calculer mAP moyen
    metrics["mAP"] = (metrics["mAP_50"] + metrics["mAP_75"]) / 2
    
    # Calculer précision et rappel pour chaque classe
    for cls_name in class_metrics:
        tp = class_metrics[cls_name]["TP"]
        fp = class_metrics[cls_name]["FP"]
        fn = class_metrics[cls_name]["FN"]
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        class_metrics[cls_name]["precision"] = precision
        class_metrics[cls_name]["recall"] = recall
        class_metrics[cls_name]["f1"] = f1
    
    metrics["class_metrics"] = class_metrics

    print("\nRésultats de l'évaluation:")
    print(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75: {metrics['mAP_75']:.4f}")
    print(f"mAP moyen: {metrics['mAP']:.4f}")
    print("\nMétriques par classe:")
    for cls_name, cls_metrics in metrics["class_metrics"].items():
        if cls_name != "DontCare":
            print(f"{cls_name}:")
            print(f"  AP: {cls_metrics['AP']:.4f}")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall: {cls_metrics['recall']:.4f}")
            print(f"  F1-score: {cls_metrics['f1']:.4f}")
    

    if save_metrics:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, f"{config['experiment_name']}_metrics.txt")
        
        with open(metrics_file, 'w') as f:
            f.write(f"mAP@0.5: {metrics['mAP_50']:.4f}\n")
            f.write(f"mAP@0.75: {metrics['mAP_75']:.4f}\n")
            f.write(f"mAP moyen: {metrics['mAP']:.4f}\n\n")
            
            f.write("Métriques par classe:\n")
            for cls_name, cls_metrics in metrics["class_metrics"].items():
                if cls_name != "DontCare":
                    f.write(f"{cls_name}:\n")
                    f.write(f"  AP: {cls_metrics['AP']:.4f}\n")
                    f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {cls_metrics['recall']:.4f}\n")
                    f.write(f"  F1-score: {cls_metrics['f1']:.4f}\n")
        
        print(f"Métriques sauvegardées dans {metrics_file}")
        plot_class_metrics(metrics, output_dir, config)
    
    return metrics

def plot_class_metrics(metrics, output_dir, config):
    """
    Crée des visualisations des métriques par classe
    
    Args:
        metrics: Dictionnaire des métriques
        output_dir: Dossier de sortie pour les visualisations
        config: Configuration de l'expérience
    """
    os.makedirs(output_dir, exist_ok=True)
    
    classes = []
    ap_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for cls_name, cls_metrics in metrics["class_metrics"].items():
        if cls_name != "DontCare":
            classes.append(cls_name)
            ap_values.append(cls_metrics["AP"])
            precision_values.append(cls_metrics["precision"])
            recall_values.append(cls_metrics["recall"])
            f1_values.append(cls_metrics["f1"])

    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.2
    
    plt.bar(x - 1.5*width, ap_values, width, label='AP')
    plt.bar(x - 0.5*width, precision_values, width, label='Precision')
    plt.bar(x + 0.5*width, recall_values, width, label='Recall')
    plt.bar(x + 1.5*width, f1_values, width, label='F1-score')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Métriques par classe')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{config['experiment_name']}_class_metrics.png"))
    plt.close()

def visualize_predictions(model, dataset, device, num_samples=5, threshold=0.5, output_dir=None, config=None):
    """
    Visualise les prédictions du modèle sur quelques échantillons
    
    Args:
        model: Modèle à utiliser pour les prédictions
        dataset: Dataset contenant les échantillons
        device: Dispositif sur lequel exécuter le modèle
        num_samples: Nombre d'échantillons à visualiser
        threshold: Seuil de confiance pour les détections
        output_dir: Dossier de sortie pour les visualisations
        config: Configuration de l'expérience
        
    Returns:
        La figure matplotlib contenant les visualisations
    """
    if config is None:
        config = EXPERIMENT_CONFIG
        
    if output_dir is None:
        output_dir = config["visualizations_dir"]
        
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            img, target = dataset[idx]

            # Préparer l'image pour le modèle
            img_tensor = img.unsqueeze(0).to(device)

            # Obtenir les prédictions
            output = model(img_tensor)[0]

            # Filtrer les détections avec un score supérieur au seuil
            keep_indices = output['scores'] > threshold
            boxes = output['boxes'][keep_indices].cpu().numpy()
            labels = output['labels'][keep_indices].cpu().numpy()
            scores = output['scores'][keep_indices].cpu().numpy()

            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
                # Dénormaliser si nécessaire
                if img_np.max() <= 1.0:
                    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np, 0, 1)
            else:
                img_np = np.array(img) / 255.0

            axes[i].imshow(img_np)

            # Dessiner les boîtes prédites
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, fill=False, edgecolor='blue', linewidth=2)
                axes[i].add_patch(rect)
                axes[i].text(x1, y1-5, f"{KITTI_CLASSES[label]}: {score:.2f}",
                             color='white', bbox=dict(facecolor='blue', alpha=0.8))

            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()

            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=2)
                axes[i].add_patch(rect)
                axes[i].text(x1, y1+height+10, KITTI_CLASSES[label],
                             color='white', bbox=dict(facecolor='red', alpha=0.8))

            axes[i].set_title(f"Échantillon {idx}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{config['experiment_name']}_predictions.png"))
    plt.close()
    
    return fig