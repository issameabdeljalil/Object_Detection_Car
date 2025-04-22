import argparse
import os
import sys
import torch

# Ajouter le dossier courant au chemin Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import EXPERIMENT_CONFIG, KITTI_CLASSES
from src.download_dataset import download_kitti_dataset
from src.train import train_model
from src.inference import inference_on_image, run_on_folder
from src.models import get_detection_model, load_checkpoint
from src.evaluate import evaluate_model, visualize_predictions
from src.data_module.data_transform import prepare_data_loaders

def main():
    parser = argparse.ArgumentParser(description='Détection d\'objets KITTI avec PyTorch')
    parser.add_argument('--mode', type=str, choices=['download', 'train', 'eval', 'visualize', 'inference', 'batch'],
                      required=True, help='Mode d\'exécution')
    parser.add_argument('--checkpoint', type=str, help='Chemin vers le checkpoint du modèle (pour eval, visualize, inference, batch)')
    parser.add_argument('--input', type=str, help='Chemin vers l\'image ou le dossier d\'images (pour inference, batch)')
    parser.add_argument('--output', type=str, help='Dossier de sortie (pour inference, batch)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Seuil de confiance (pour inference, batch)')
    parser.add_argument('--num_samples', type=int, default=5, help='Nombre d\'échantillons à visualiser (pour visualize)')
    
    args = parser.parse_args()
    
    # GPU ou CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositif utilisé: {device}")
    
    if args.mode == 'download':
        data_dir = download_kitti_dataset(EXPERIMENT_CONFIG["data_dir"])
        print(f"Dataset KITTI téléchargé à {data_dir}")
    
    elif args.mode == 'train':
        model, train_losses, val_metrics = train_model(EXPERIMENT_CONFIG)
        print("Entraînement terminé")
    
    elif args.mode == 'eval':
        if not args.checkpoint:
            print("Erreur: Le paramètre --checkpoint est requis pour le mode 'eval'")
            return
        
        data_dir = download_kitti_dataset(EXPERIMENT_CONFIG["data_dir"])
        
        dataset, train_dataset, val_dataset, train_loader, val_loader = prepare_data_loaders(
            data_dir, 
            img_size=EXPERIMENT_CONFIG["img_size"], 
            batch_size=EXPERIMENT_CONFIG["batch_size"]
        )
        
        model = get_detection_model(num_classes=len(KITTI_CLASSES))
        model, _, _, _, _, _ = load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)
        metrics = evaluate_model(model, val_loader, device, save_metrics=True)
    
    elif args.mode == 'visualize':
        if not args.checkpoint:
            print("Erreur: Le paramètre --checkpoint est requis pour le mode 'visualize'")
            return
        data_dir = download_kitti_dataset(EXPERIMENT_CONFIG["data_dir"])
        
        # Préparer les données
        dataset, train_dataset, val_dataset, train_loader, val_loader = prepare_data_loaders(
            data_dir, 
            img_size=EXPERIMENT_CONFIG["img_size"], 
            batch_size=EXPERIMENT_CONFIG["batch_size"]
        )
        model = get_detection_model(num_classes=len(KITTI_CLASSES))
        model, _, _, _, _, _ = load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)
        
        # Graphiques
        visualize_predictions(model, val_dataset, device, num_samples=args.num_samples, threshold=args.threshold)
        print(f"Visualisations sauvegardées dans {EXPERIMENT_CONFIG['visualizations_dir']}")
    
    elif args.mode == 'inference':
        # Inférence sur une image
        if not args.checkpoint or not args.input:
            print("Erreur: Les paramètres --checkpoint et --input sont requis pour le mode 'inference'")
            return
            
        # Initialiser le modèle
        model = get_detection_model(num_classes=len(KITTI_CLASSES))
        model, _, _, _, _, _ = load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)
        output_dir = args.output if args.output else EXPERIMENT_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Inférence sur l'image
        original_image, result_image, detections = inference_on_image(
            model, args.input, device, args.threshold, EXPERIMENT_CONFIG["img_size"]
        )
        print("\nDétections:")
        for i, detection in enumerate(detections):
            print(f"{i+1}. {detection['class']} ({detection['confidence']:.2f}): {detection['bbox']}")
        
        
        output_path = os.path.join(output_dir, os.path.basename(args.input))
        result_image.save(output_path)
        print(f"\nImage avec détections sauvegardée dans: {output_path}")
    
    elif args.mode == 'batch':
        # Inférence sur un dossier d'images
        if not args.checkpoint or not args.input:
            print("Erreur: Les paramètres --checkpoint et --input sont requis pour le mode 'batch'")
            return
            
        # Initialiser le modèle
        model = get_detection_model(num_classes=len(KITTI_CLASSES))
        model, _, _, _, _, _ = load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)
        
        # Définir le dossier de sortie s'il n'est pas spécifié
        output_dir = args.output if args.output else os.path.join(EXPERIMENT_CONFIG["output_dir"], "batch_results")
        
        # Exécuter l'inférence sur le dossier
        run_on_folder(
            model, args.input, output_dir, device, args.threshold, EXPERIMENT_CONFIG["img_size"]
        )

if __name__ == "__main__":
    main()