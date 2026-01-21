#!/usr/bin/env python3
"""
Predicción con PointNet++ entrenado
Predice la clase de nuevas estructuras atómicas
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from pointnet2_simple import create_classifier


def read_off(filename):
    """Lee archivo OFF"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if lines[0].strip() != 'OFF':
        raise ValueError(f"Formato OFF inválido en {filename}")
    
    n_vertices = int(lines[1].strip().split()[0])
    
    vertices = []
    for i in range(2, 2 + n_vertices):
        coords = [float(x) for x in lines[i].strip().split()]
        vertices.append(coords[:3])
    
    return np.array(vertices, dtype=np.float32)


def preprocess_structure(point_cloud, n_points=1024):
    """Preprocesa estructura para predicción"""
    # Samplear puntos
    if len(point_cloud) >= n_points:
        choice = np.random.choice(len(point_cloud), n_points, replace=False)
    else:
        choice = np.random.choice(len(point_cloud), n_points, replace=True)
    
    point_cloud = point_cloud[choice, :]
    
    # Normalizar
    point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    max_dist = np.max(np.linalg.norm(point_cloud, axis=1))
    if max_dist > 0:
        point_cloud = point_cloud / max_dist
    
    # Formato PyTorch [1, 3, N]
    point_cloud = torch.from_numpy(point_cloud).float().transpose(0, 1).unsqueeze(0)
    
    return point_cloud


def predict_single(model, structure_file, n_points, class_names, device):
    """Predice clase de una estructura"""
    # Cargar estructura
    point_cloud = read_off(structure_file)
    
    # Preprocesar
    input_tensor = preprocess_structure(point_cloud, n_points=n_points)
    input_tensor = input_tensor.to(device)
    
    # Predecir
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[predicted_class].item()
    
    return {
        'class_idx': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {class_names[i]: probabilities[i].item() for i in range(len(class_names))}
    }


def main():
    parser = argparse.ArgumentParser(description='Predecir clase de estructuras atómicas')
    parser.add_argument('--model', type=str, required=True, help='Path al modelo (.pth)')
    parser.add_argument('--input', type=str, required=True, help='Archivo .off o directorio')
    parser.add_argument('--batch', action='store_true', help='Procesar directorio completo')
    
    args = parser.parse_args()
    
    # Cargar modelo
    print("Cargando modelo...")
    checkpoint = torch.load(args.model, map_location='cpu')
    
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    n_points = checkpoint['n_points']
    
    model = create_classifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cpu')
    model = model.to(device)
    
    print(f"Modelo cargado:")
    print(f"  Clases: {class_names}")
    print(f"  Puntos: {n_points}")
    print(f"  Accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
    print()
    
    # Predicción
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Procesar directorio
        print(f"Procesando directorio: {input_path}")
        off_files = list(input_path.glob('*.off'))
        
        if not off_files:
            print("❌ No se encontraron archivos .off")
            return
        
        print(f"Encontrados {len(off_files)} archivos\n")
        
        results = []
        for off_file in off_files:
            try:
                result = predict_single(model, off_file, n_points, class_names, device)
                result['file'] = off_file.name
                results.append(result)
                
                print(f"📄 {off_file.name}")
                print(f"   Clase: {result['class_name']}")
                print(f"   Confianza: {result['confidence']*100:.2f}%")
                print()
            except Exception as e:
                print(f"❌ Error procesando {off_file.name}: {e}\n")
        
        # Resumen
        print("="*70)
        print("RESUMEN")
        print("="*70)
        for class_name in class_names:
            count = sum(1 for r in results if r['class_name'] == class_name)
            print(f"{class_name}: {count} estructuras")
        
    else:
        # Predicción individual
        print(f"Procesando: {input_path.name}\n")
        
        try:
            result = predict_single(model, input_path, n_points, class_names, device)
            
            print("="*70)
            print("RESULTADO")
            print("="*70)
            print(f"Archivo: {input_path.name}")
            print(f"Clase predicha: {result['class_name']}")
            print(f"Confianza: {result['confidence']*100:.2f}%")
            print()
            print("Probabilidades por clase:")
            for class_name, prob in result['probabilities'].items():
                bar = '█' * int(prob * 50)
                print(f"  {class_name:15s} {prob*100:5.2f}% {bar}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
