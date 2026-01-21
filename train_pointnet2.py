#!/usr/bin/env python3
"""
Entrenamiento de PointNet++ para Estructuras Atómicas de LAMMPS
Versión simplificada - funciona en CPU

Usa archivos .off generados por dump_to_off.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from pointnet2_simple import create_classifier


class AtomicStructureDataset(Dataset):
    """
    Dataset para estructuras atómicas en formato OFF
    
    Estructura de carpetas esperada:
    data/
      train/
        clase_0/
          estructura_001.off
          estructura_002.off
        clase_1/
          estructura_001.off
      test/
        clase_0/
        clase_1/
    """
    
    def __init__(self, root_dir, n_points=1024, split='train', augment=False):
        self.root_dir = Path(root_dir) / split
        self.n_points = n_points
        self.augment = augment
        
        # Cargar archivos
        self.files = []
        self.labels = []
        
        classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_names = [c.name for c in classes]
        self.num_classes = len(classes)
        
        for class_idx, class_dir in enumerate(classes):
            off_files = list(class_dir.glob('*.off'))
            self.files.extend(off_files)
            self.labels.extend([class_idx] * len(off_files))
        
        print(f"Dataset {split}:")
        print(f"  Clases: {self.class_names}")
        print(f"  Total archivos: {len(self.files)}")
        for i, name in enumerate(self.class_names):
            count = sum(1 for l in self.labels if l == i)
            print(f"    - {name}: {count} estructuras")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Leer archivo OFF
        point_cloud = self.read_off(self.files[idx])
        label = self.labels[idx]
        
        # Samplear puntos
        if len(point_cloud) >= self.n_points:
            choice = np.random.choice(len(point_cloud), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(point_cloud), self.n_points, replace=True)
        
        point_cloud = point_cloud[choice, :]
        
        # Normalizar (centrar y escalar)
        point_cloud = point_cloud - np.mean(point_cloud, axis=0)
        max_dist = np.max(np.linalg.norm(point_cloud, axis=1))
        if max_dist > 0:
            point_cloud = point_cloud / max_dist
        
        # Augmentation (rotación aleatoria)
        if self.augment:
            point_cloud = self.rotate_point_cloud(point_cloud)
        
        # Formato PyTorch [3, N]
        point_cloud = torch.from_numpy(point_cloud).float().transpose(0, 1)
        label = torch.tensor(label).long()
        
        return point_cloud, label
    
    @staticmethod
    def read_off(filename):
        """Lee archivo OFF y retorna coordenadas"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Primera línea debe ser "OFF"
        if lines[0].strip() != 'OFF':
            raise ValueError(f"Formato OFF inválido en {filename}")
        
        # Segunda línea: n_vertices n_faces n_edges
        n_vertices = int(lines[1].strip().split()[0])
        
        # Leer vértices
        vertices = []
        for i in range(2, 2 + n_vertices):
            coords = [float(x) for x in lines[i].strip().split()]
            vertices.append(coords[:3])
        
        return np.array(vertices, dtype=np.float32)
    
    @staticmethod
    def rotate_point_cloud(point_cloud):
        """Rotación aleatoria para augmentation"""
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                   [0, 1, 0],
                                   [-sinval, 0, cosval]])
        rotated = np.dot(point_cloud, rotation_matrix)
        return rotated


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena por una época"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        
        # Verificar NaN en loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  Training batch con NaN/Inf loss, saltando...")
            continue
        
        loss.backward()
        
        # Gradient clipping para evitar explosión
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Valida el modelo con protección contra NaN/Inf"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    skipped_batches = 0
    
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(tqdm(dataloader, desc='Validation')):
            points = points.to(device)
            labels = labels.to(device)
            
            # Verificar NaN/Inf en input
            if torch.isnan(points).any() or torch.isinf(points).any():
                print(f"\n⚠️  Batch {batch_idx}: Input con NaN/Inf, saltando...")
                skipped_batches += 1
                continue
            
            try:
                outputs = model(points)
                
                # Verificar NaN/Inf en output
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\n⚠️  Batch {batch_idx}: Output con NaN/Inf, saltando...")
                    skipped_batches += 1
                    continue
                
                loss = criterion(outputs, labels)
                
                # Verificar loss válido
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    print(f"\n⚠️  Batch {batch_idx}: Loss inválido ({loss.item():.2f}), saltando...")
                    skipped_batches += 1
                    continue
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"\n⚠️  Batch {batch_idx}: Error {e}, saltando...")
                skipped_batches += 1
                continue
    
    if skipped_batches > 0:
        print(f"\n⚠️  Se saltaron {skipped_batches} batches problemáticos")
    
    # Calcular promedios solo con batches válidos
    valid_batches = len(dataloader) - skipped_batches
    
    if valid_batches == 0:
        print("\n❌ ERROR: Todos los batches fueron saltados!")
        return float('inf'), 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = 100 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Entrenar PointNet++ para estructuras atómicas')
    parser.add_argument('--data', type=str, required=True, help='Directorio de datos')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')  # Reducido de 0.001
    parser.add_argument('--n_points', type=int, default=1024, help='Puntos por estructura')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directorio para guardar modelos')
    parser.add_argument('--augment', action='store_true', help='Usar data augmentation')
    
    args = parser.parse_args()
    
    # Crear directorio para checkpoints
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cpu')
    print(f"\nUsando: {device}")
    
    # Datasets
    print("\nCargando datasets...")
    train_dataset = AtomicStructureDataset(
        args.data, 
        n_points=args.n_points, 
        split='train',
        augment=args.augment
    )
    test_dataset = AtomicStructureDataset(
        args.data, 
        n_points=args.n_points, 
        split='test',
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Modelo
    print(f"\nCreando modelo para {train_dataset.num_classes} clases...")
    model = create_classifier(num_classes=train_dataset.num_classes)
    model = model.to(device)
    
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento por {args.epochs} épocas...\n")
    
    best_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Época {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        # Scheduler
        scheduler.step()
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Imprimir resultados
        print(f"\nResultados Época {epoch+1}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # Guardar mejor modelo
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'class_names': train_dataset.class_names,
                'num_classes': train_dataset.num_classes,
                'n_points': args.n_points
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"  ✅ Mejor modelo guardado! (Acc: {best_acc:.2f}%)")
    
    # Guardar modelo final
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'class_names': train_dataset.class_names,
        'num_classes': train_dataset.num_classes,
        'n_points': args.n_points
    }
    torch.save(final_checkpoint, save_dir / 'final_model.pth')
    
    # Guardar historial
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Entrenamiento completado!")
    print(f"{'='*70}")
    print(f"Mejor accuracy: {best_acc:.2f}%")
    print(f"Modelos guardados en: {save_dir}/")
    print(f"  - best_model.pth (mejor accuracy)")
    print(f"  - final_model.pth (última época)")
    print(f"  - history.json (métricas)")


if __name__ == '__main__':
    main()
