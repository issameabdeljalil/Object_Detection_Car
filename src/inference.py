import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import json

from src.config import KITTI_CLASSES, EXPERIMENT_CONFIG
from src.models import get_detection_model, load_checkpoint

def inference_on_image(model, image_path, device, threshold=0.5, img_size=(375, 1242)):
    """
    Effectue l'inférence sur une seule image
    
    Args:
        model: Le modèle de détection
        image_path: Chemin vers l'image à traiter
        device: Dispositif (cuda ou cpu)
        threshold: Seuil de confiance pour les détections (0-1)
        img_size: Taille de redimensionnement de l'image
        
    Returns:
        Tuple: (image originale, image avec détections, liste des détections)
    """
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    
    # Appliquer les transformations
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformed_image = transform(original_image)
    
    # Envoyer l'image au modèle
    model.eval()
    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0).to(device)
        predictions = model(transformed_image)[0]
    
    # Filtrer les prédictions avec un score supérieur au seuil
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    mask = scores >= threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Créer une copie de l'image pour dessiner les détections
    draw_image = original_image.copy()
    draw = cv2.cvtColor(np.array(draw_image), cv2.COLOR_RGB2BGR)
    detections = []
    
    # Dessiner les boîtes et les étiquettes
    for box, label, score in zip(boxes, labels, scores):
        # Convertir les coordonnées de boîte
        x1, y1, x2, y2 = box.astype(int)
        
        scale_w = width / img_size[1]
        scale_h = height / img_size[0]
        
        x1 = int(x1 * scale_w)
        y1 = int(y1 * scale_h)
        x2 = int(x2 * scale_w)
        y2 = int(y2 * scale_h)
        
        class_name = KITTI_CLASSES[label]
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': [x1, y1, x2, y2]
        })
        
        # Dessiner la boîte
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Dessiner l'étiquette
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(draw, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convertir le dessin en PIL Image pour la cohérence du retour
    result_image = Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
    
    return original_image, result_image, detections

def run_on_folder(model, folder_path, output_folder, device, threshold=0.5, img_size=(375, 1242)):
    """
    Exécute le modèle sur toutes les images d'un dossier
    
    Args:
        model: Le modèle de détection
        folder_path: Chemin vers le dossier contenant les images
        output_folder: Dossier de sortie pour les images avec détections
        device: Dispositif (cuda ou cpu)
        threshold: Seuil de confiance pour les détections (0-1)
        img_size: Taille de redimensionnement des images
    """
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f)) and 
                   any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"Aucune image trouvée dans {folder_path}")
        return
    
    print(f"Traitement de {len(image_files)} images...")
    
    results = []
    
    from tqdm import tqdm
    for img_file in tqdm(image_files):
        img_path = os.path.join(folder_path, img_file)
        
        try:
            _, result_image, detections = inference_on_image(model, img_path, device, threshold, img_size)
            
            output_path = os.path.join(output_folder, img_file)
            result_image.save(output_path)
            results.append({
                'image': img_file,
                'detections': detections
            })
            
        except Exception as e:
            print(f"Erreur lors du traitement de {img_file}: {e}")
    
    # Sauvegarder les résultats au format JSON
    with open(os.path.join(output_folder, 'detections.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Traitement terminé. Résultats sauvegardés dans {output_folder}")
    
    # Afficher quelques statistiques
    total_detections = sum(len(r['detections']) for r in results)
    class_counts = {}
    
    for result in results:
        for detection in result['detections']:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"Total des détections: {total_detections}")
    print("Détections par classe:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inférence avec le modèle de détection d\'objets KITTI')
    parser.add_argument('--checkpoint', type=str, required=True, help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--mode', type=str, choices=['image', 'folder'], required=True, help='Mode d\'inférence: image unique ou dossier')
    parser.add_argument('--input', type=str, required=True, help='Chemin vers l\'image ou le dossier d\'images')
    parser.add_argument('--output', type=str, default='./outputs/inference', help='Dossier de sortie pour les résultats')
    parser.add_argument('--threshold', type=float, default=0.5, help='Seuil de confiance pour les détections (0-1)')
    
    args = parser.parse_args()
    
    # GPU ou CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositif utilisé: {device}")
    
    model = get_detection_model(num_classes=len(KITTI_CLASSES))
    model, _, _, _, _, _ = load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    
    if args.mode == 'image':
        # Inférence sur une seule image
        print(f"Traitement de l'image {args.input}...")
        os.makedirs(args.output, exist_ok=True)
        
        original_image, result_image, detections = inference_on_image(model, args.input, device, args.threshold)
        
        print("\nDétections:")
        for i, detection in enumerate(detections):
            print(f"{i+1}. {detection['class']} ({detection['confidence']:.2f}): {detection['bbox']}")
        output_path = os.path.join(args.output, os.path.basename(args.input))
        result_image.save(output_path)
        print(f"\nImage avec détections sauvegardée dans: {output_path}")
        
        # Sauvegarder les détections en JSON
        with open(os.path.join(args.output, "detections.json"), 'w') as f:
            json.dump(detections, f, indent=2)
            
    elif args.mode == 'folder':
        # Inférence sur un dossier d'images
        print(f"Traitement du dossier {args.input}...")
        run_on_folder(model, args.input, args.output, device, args.threshold)
    
if __name__ == "__main__":
    main()