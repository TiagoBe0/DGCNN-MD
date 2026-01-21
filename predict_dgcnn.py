#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicción con DGCNN entrenado
==============================

Uso:
    # Predecir un archivo
    python predict_dgcnn.py --model modelo_dgcnn --input archivo.off
    
    # Predecir un directorio
    python predict_dgcnn.py --model modelo_dgcnn --input directorio/
    
    # Predecir desde dump de LAMMPS
    python predict_dgcnn.py --model modelo_dgcnn --input archivo.dump --from_dump
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DGCNN MODEL (mismo que en train_dgcnn.py)
# =============================================================================

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
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
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNN(nn.Module):
    def __init__(self, num_classes=4, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.conv1 = EdgeConv(3, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 128, k=k)
        self.conv4 = EdgeConv(128, 256, k=k)
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
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
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x_max, x_avg), dim=1)
        
        x = self.classifier(x)
        return x


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_off(filepath, n_points=256):
    """Carga archivo OFF"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if lines[0].strip() == 'OFF':
        n_verts = int(lines[1].split()[0])
        start = 2
    else:
        parts = lines[0].split()
        n_verts = int(parts[1]) if len(parts) > 1 else int(lines[1].split()[0])
        start = 1 if len(parts) > 1 else 2
    
    points = []
    for i in range(start, start + n_verts):
        if i < len(lines):
            coords = [float(x) for x in lines[i].split()[:3]]
            points.append(coords)
    
    points = np.array(points, dtype=np.float32)
    
    # Ajustar número de puntos
    n = len(points)
    if n >= n_points:
        choice = np.random.choice(n, n_points, replace=False)
    else:
        choice = np.random.choice(n, n_points, replace=True)
    
    points = points[choice, :]
    
    # Normalizar
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    return points


def load_dump_surface(filepath, n_points=256):
    """
    Carga archivo dump de LAMMPS y extrae superficie usando OVITO
    """
    try:
        from ovito.io import import_file
        from ovito.modifiers import ConstructSurfaceModifier, InvertSelectionModifier, DeleteSelectedModifier
    except ImportError:
        print("❌ Error: OVITO no está instalado")
        print("   Instalar con: pip install ovito --break-system-packages")
        return None
    
    pipeline = import_file(filepath)
    
    pipeline.modifiers.append(ConstructSurfaceModifier(
        radius=2.0,
        select_surface_particles=True,
        smoothing_level=12
    ))
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    data = pipeline.compute()
    points = np.array(data.particles['Position'][...], dtype=np.float32)
    
    if len(points) < 4:
        return None
    
    # Ajustar número de puntos
    n = len(points)
    if n >= n_points:
        choice = np.random.choice(n, n_points, replace=False)
    else:
        choice = np.random.choice(n, n_points, replace=True)
    
    points = points[choice, :]
    
    # Normalizar
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    return points


def load_model(model_dir):
    """Carga modelo entrenado"""
    model_dir = Path(model_dir)
    
    # Cargar config
    config_path = model_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        classes = config.get('classes', ['single_vacancy', 'small_cluster', 'medium_cluster', 'large_cluster'])
        k = config.get('k', 20)
        dropout = config.get('dropout', 0.5)
        n_points = config.get('n_points', 256)
    else:
        classes = ['single_vacancy', 'small_cluster', 'medium_cluster', 'large_cluster']
        k = 20
        dropout = 0.5
        n_points = 256
    
    # Cargar modelo
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=DEVICE)
    
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
    
    model = DGCNN(num_classes=len(classes), k=k, dropout=dropout).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, classes, n_points


def predict(model, points, classes):
    """Realiza predicción"""
    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(points_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    return classes[pred_idx], confidence, probs[0].cpu().numpy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predicción con DGCNN')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Directorio del modelo entrenado')
    parser.add_argument('--input', type=str, required=True,
                        help='Archivo .off o directorio a predecir')
    parser.add_argument('--from_dump', action='store_true',
                        help='Entrada es archivo(s) dump de LAMMPS')
    parser.add_argument('--output', type=str, default=None,
                        help='Archivo CSV de salida (opcional)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔷 DGCNN - Predicción")
    print("=" * 60)
    
    # Cargar modelo
    print(f"Cargando modelo desde: {args.model}")
    model, classes, n_points = load_model(args.model)
    print(f"Clases: {classes}")
    print(f"Puntos: {n_points}")
    
    # Determinar archivos a predecir
    input_path = Path(args.input)
    
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        if args.from_dump:
            files = list(input_path.glob('*')) + list(input_path.glob('*.dump'))
        else:
            files = list(input_path.glob('*.off'))
        files = sorted(files)
    else:
        print(f"❌ Error: {args.input} no existe")
        return
    
    print(f"\nArchivos a predecir: {len(files)}")
    print("-" * 60)
    
    results = []
    
    for filepath in files:
        # Cargar puntos
        if args.from_dump:
            points = load_dump_surface(filepath, n_points)
        else:
            points = load_off(filepath, n_points)
        
        if points is None:
            print(f"⚠️  {filepath.name}: no se pudo cargar")
            continue
        
        # Predecir
        pred_class, confidence, probs = predict(model, points, classes)
        
        # Mostrar resultado
        print(f"{filepath.name:40} → {pred_class:20} ({confidence*100:5.1f}%)")
        
        results.append({
            'file': filepath.name,
            'prediction': pred_class,
            'confidence': confidence,
            **{f'prob_{c}': p for c, p in zip(classes, probs)}
        })
    
    # Guardar resultados
    if args.output and results:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Resultados guardados en: {args.output}")
    
    print("\n" + "=" * 60)
    print("✅ PREDICCIÓN COMPLETADA")
    print("=" * 60)


if __name__ == '__main__':
    main()
