#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGCNN (Dynamic Graph CNN) para Clasificación de Clusters de Vacancias
=====================================================================

Arquitectura que construye grafos dinámicos entre puntos vecinos,
capturando mejor la estructura local que PointNet++.

Referencia: "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., 2019)

Uso:
    python train_dgcnn.py --data data_pointnet --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Verificar si tenemos GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATASET
# =============================================================================

class VacancyClusterDataset(Dataset):
    """
    Dataset para archivos .off de clusters de vacancias
    
    Estructura esperada:
        data_dir/
            train/
                single_vacancy/
                small_cluster/
                medium_cluster/
                large_cluster/
            test/
                ...
    """
    
    def __init__(self, root_dir, split='train', n_points=256, augment=False):
        self.root_dir = Path(root_dir) / split
        self.n_points = n_points
        self.augment = augment
        
        # Detectar clases
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Recolectar archivos
        self.files = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for off_file in class_dir.glob('*.off'):
                self.files.append(off_file)
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Dataset {split}: {len(self.files)} archivos, {len(self.classes)} clases")
        for c in self.classes:
            count = sum(1 for l in self.labels if l == self.class_to_idx[c])
            print(f"  - {c}: {count}")
    
    def __len__(self):
        return len(self.files)
    
    def load_off(self, filepath):
        """Carga archivo OFF y retorna coordenadas"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Saltar header "OFF"
        if lines[0].strip() == 'OFF':
            n_verts = int(lines[1].split()[0])
            start = 2
        else:
            # Formato "OFF n_verts n_faces n_edges"
            parts = lines[0].split()
            n_verts = int(parts[1]) if len(parts) > 1 else int(lines[1].split()[0])
            start = 1 if len(parts) > 1 else 2
        
        # Leer vértices
        points = []
        for i in range(start, start + n_verts):
            if i < len(lines):
                coords = [float(x) for x in lines[i].split()[:3]]
                points.append(coords)
        
        return np.array(points, dtype=np.float32)
    
    def augment_pointcloud(self, points):
        """Data augmentation para point clouds"""
        
        # 1. Rotación aleatoria alrededor del eje Z
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_z = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points = points @ rotation_z.T
        
        # 2. Rotación aleatoria alrededor del eje Y (menos agresiva)
        theta_y = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 grados
        cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
        rotation_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=np.float32)
        points = points @ rotation_y.T
        
        # 3. Jitter (ruido gaussiano pequeño)
        noise = np.random.normal(0, 0.02, points.shape).astype(np.float32)
        points = points + noise
        
        # 4. Random scaling
        scale = np.random.uniform(0.9, 1.1)
        points = points * scale
        
        # 5. Random translation pequeña
        translation = np.random.uniform(-0.1, 0.1, (1, 3)).astype(np.float32)
        points = points + translation
        
        return points
    
    def __getitem__(self, idx):
        # Cargar puntos
        points = self.load_off(self.files[idx])
        
        # Asegurar n_points
        n = len(points)
        if n >= self.n_points:
            # Submuestreo aleatorio
            choice = np.random.choice(n, self.n_points, replace=False)
        else:
            # Upsample con reemplazo
            choice = np.random.choice(n, self.n_points, replace=True)
        
        points = points[choice, :]
        
        # Augmentation
        if self.augment:
            points = self.augment_pointcloud(points)
        
        # Normalizar (centrar y escalar)
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Convertir a tensor
        points = torch.from_numpy(points).float()
        label = torch.tensor(self.labels[idx]).long()
        
        return points, label


# =============================================================================
# DGCNN MODEL
# =============================================================================

def knn(x, k):
    """
    Encuentra k vecinos más cercanos
    
    Args:
        x: tensor (batch_size, num_dims, num_points)
        k: número de vecinos
    
    Returns:
        idx: índices de vecinos (batch_size, num_points, k)
    """
    # Calcular distancias pairwise
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    
    # Obtener k vecinos más cercanos
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construye features de grafo para EdgeConv
    
    Args:
        x: tensor (batch_size, num_dims, num_points)
        k: número de vecinos
        idx: índices pre-computados (opcional)
    
    Returns:
        feature: tensor (batch_size, 2*num_dims, num_points, k)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Concatenar: [punto central, diferencia con vecinos]
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature


class EdgeConv(nn.Module):
    """EdgeConv layer para DGCNN"""
    
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        # x: (B, C, N)
        x = get_graph_feature(x, k=self.k)  # (B, 2C, N, k)
        x = self.conv(x)  # (B, out_channels, N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, out_channels, N)
        return x


class DGCNN(nn.Module):
    """
    Dynamic Graph CNN para clasificación de point clouds
    
    Arquitectura:
        EdgeConv (3 -> 64) -> EdgeConv (64 -> 64) -> EdgeConv (64 -> 128) -> EdgeConv (128 -> 256)
        -> Global Max Pool + Global Avg Pool
        -> MLP (512 -> 256 -> num_classes)
    """
    
    def __init__(self, num_classes=4, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        
        # EdgeConv layers
        self.conv1 = EdgeConv(3, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 128, k=k)
        self.conv4 = EdgeConv(128, 256, k=k)
        
        # Aggregation MLP
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        
        # EdgeConv blocks
        x1 = self.conv1(x)    # (B, 64, N)
        x2 = self.conv2(x1)   # (B, 64, N)
        x3 = self.conv3(x2)   # (B, 128, N)
        x4 = self.conv4(x3)   # (B, 256, N)
        
        # Concatenar features de todas las escalas
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # MLP adicional
        x = self.conv5(x)  # (B, 1024, N)
        
        # Global pooling (max + avg)
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (B, 1024)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (B, 1024)
        x = torch.cat((x_max, x_avg), dim=1)  # (B, 2048)
        
        # Clasificador
        x = self.classifier(x)
        
        return x


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """Early stopping para evitar overfitting"""
    
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def calculate_class_weights(dataset):
    """Calcula pesos para balancear clases desbalanceadas"""
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * points.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evalúa el modelo"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * points.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / total, 100. * correct / total, all_preds, all_labels


def print_confusion_matrix(preds, labels, class_names):
    """Imprime matriz de confusión"""
    from collections import defaultdict
    
    n_classes = len(class_names)
    matrix = defaultdict(lambda: defaultdict(int))
    
    for pred, label in zip(preds, labels):
        matrix[label][pred] += 1
    
    print("\n📊 Matriz de Confusión:")
    print("-" * 60)
    
    # Header
    header = "Real \\ Pred".ljust(18)
    for i, name in enumerate(class_names):
        header += f"{name[:8]:>10}"
    print(header)
    print("-" * 60)
    
    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:16]:16}"
        for j in range(n_classes):
            row += f"{matrix[i][j]:>10}"
        print(row)
    
    print("-" * 60)
    
    # Per-class accuracy
    print("\n📈 Accuracy por clase:")
    for i, name in enumerate(class_names):
        total = sum(matrix[i].values())
        correct = matrix[i][i]
        acc = 100 * correct / total if total > 0 else 0
        print(f"  {name:20}: {acc:6.2f}% ({correct}/{total})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DGCNN para clasificación de clusters de vacancias'
    )
    
    parser.add_argument('--data', type=str, required=True,
                        help='Directorio de datos (con train/ y test/)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--n_points', type=int, default=256,
                        help='Puntos por muestra (default: 256)')
    parser.add_argument('--k', type=int, default=20,
                        help='Número de vecinos para grafo (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--save_dir', type=str, default='modelo_dgcnn',
                        help='Directorio para guardar modelo')
    parser.add_argument('--no_augment', action='store_true',
                        help='Desactivar data augmentation')
    parser.add_argument('--patience', type=int, default=20,
                        help='Paciencia para early stopping (default: 20)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔷 DGCNN - Dynamic Graph CNN para Clusters de Vacancias")
    print("=" * 70)
    print(f"Dispositivo: {DEVICE}")
    print(f"Datos: {args.data}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Puntos por muestra: {args.n_points}")
    print(f"K vecinos: {args.k}")
    print(f"Dropout: {args.dropout}")
    print(f"Data augmentation: {'No' if args.no_augment else 'Sí'}")
    print("=" * 70)
    
    # Crear directorio de salida
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datasets
    print("\n📂 Cargando datasets...")
    train_dataset = VacancyClusterDataset(
        args.data, split='train', 
        n_points=args.n_points, 
        augment=not args.no_augment
    )
    test_dataset = VacancyClusterDataset(
        args.data, split='test', 
        n_points=args.n_points, 
        augment=False
    )
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("❌ Error: datasets vacíos")
        return
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Modelo
    num_classes = len(train_dataset.classes)
    model = DGCNN(num_classes=num_classes, k=args.k, dropout=args.dropout).to(DEVICE)
    
    # Contar parámetros
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Modelo DGCNN: {n_params:,} parámetros")
    
    # Loss con pesos de clase
    class_weights = calculate_class_weights(train_dataset).to(DEVICE)
    print(f"📊 Pesos de clase: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    print("\n" + "=" * 70)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Evaluate
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print progress
        print(f"Época {epoch:3d}/{args.epochs} | "
              f"Train: {train_acc:6.2f}% (loss: {train_loss:.4f}) | "
              f"Test: {test_acc:6.2f}% (loss: {test_loss:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}", end="")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'classes': train_dataset.classes
            }, save_dir / 'best_model.pth')
            print(f" ✅ Mejor modelo!")
        else:
            print()
        
        # Early stopping
        if early_stopping(test_acc):
            print(f"\n⏹️  Early stopping en época {epoch}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("📊 EVALUACIÓN FINAL")
    print("=" * 70)
    
    # Cargar mejor modelo
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\n✅ Mejor accuracy en test: {best_acc:.2f}%")
    print(f"   (Época {checkpoint['epoch']})")
    
    # Confusion matrix
    print_confusion_matrix(preds, labels, train_dataset.classes)
    
    # Guardar historial
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Guardar config
    config = vars(args)
    config['best_acc'] = best_acc
    config['best_epoch'] = checkpoint['epoch']
    config['classes'] = train_dataset.classes
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📁 Modelo guardado en: {save_dir}/")
    print("   - best_model.pth")
    print("   - history.json")
    print("   - config.json")
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
