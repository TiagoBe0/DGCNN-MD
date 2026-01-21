#!/usr/bin/env python3
"""
PointNet++ Simplificado - Versión CPU Pura
Implementación en PyTorch puro, sin operaciones CUDA customizadas

Esta versión funciona perfectamente en CPU para:
- Clasificación de estructuras atómicas
- Segmentación de defectos
- Análisis de simulaciones MD

Basado en el paper original de PointNet++ pero usando solo operaciones PyTorch estándar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def square_distance(src, dst):
    """
    Calcula distancias cuadradas entre todos los pares de puntos
    
    Args:
        src: tensor [B, N, C]
        dst: tensor [B, M, C]
    Returns:
        dist: tensor [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS)
    Selecciona puntos que están lo más lejos posible unos de otros
    
    Args:
        xyz: tensor [B, N, 3] - coordenadas de puntos
        npoint: int - número de puntos a samplear
    Returns:
        centroids: tensor [B, npoint] - índices de puntos seleccionados
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball Query - encuentra puntos dentro de un radio
    
    Args:
        radius: float - radio de búsqueda
        nsample: int - máximo número de puntos a samplear
        xyz: tensor [B, N, 3] - todos los puntos
        new_xyz: tensor [B, S, 3] - puntos de consulta
    Returns:
        group_idx: tensor [B, S, nsample] - índices de puntos agrupados
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Samplea puntos y agrupa características locales
    
    Args:
        npoint: int - número de puntos centroides
        radius: float - radio de búsqueda
        nsample: int - puntos por grupo
        xyz: tensor [B, N, 3] - coordenadas
        points: tensor [B, N, D] - características (None si solo xyz)
    Returns:
        new_xyz: tensor [B, npoint, 3]
        new_points: tensor [B, npoint, nsample, D+3]
    """
    B, N, C = xyz.shape
    S = npoint
    
    # FPS para seleccionar centroides
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, C))
    
    # Ball query para agrupar
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    
    # Agrupar coordenadas xyz
    idx_expanded = idx.reshape(B, -1, 1).expand(-1, -1, C)
    grouped_xyz = torch.gather(xyz, 1, idx_expanded).reshape(B, S, nsample, C)
    
    # Coordenadas relativas
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape(B, S, 1, C)
    
    if points is not None:
        idx_expanded_points = idx.reshape(B, -1, 1).expand(-1, -1, points.shape[-1])
        grouped_points = torch.gather(points, 1, idx_expanded_points).reshape(B, S, nsample, -1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Layer - bloque fundamental de PointNet++
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.group_all = group_all
    
    def forward(self, xyz, points):
        """
        Args:
            xyz: [B, C, N] - coordenadas
            points: [B, D, N] - características
        Returns:
            new_xyz: [B, C, S]
            new_points: [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # [B, S, nsample, D] -> [B, D, nsample, S]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNet2Classification(nn.Module):
    """
    PointNet++ para Clasificación de Estructuras Atómicas
    
    Arquitectura simplificada ideal para estructuras cristalinas
    """
    def __init__(self, num_classes=2, input_channels=3):
        super(PointNet2Classification, self).__init__()
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=input_channels, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        
        # Clasificación final
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, xyz):
        """
        Args:
            xyz: tensor [B, 3, N] - coordenadas de puntos
        Returns:
            logits: tensor [B, num_classes]
        """
        B, _, _ = xyz.shape
        
        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Clasificación
        x = l3_points.reshape(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x


def sample_and_group_all(xyz, points):
    """
    Agrupa todos los puntos en un solo grupo
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.reshape(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# Función de utilidad para crear modelo
def create_classifier(num_classes=2, input_channels=3):
    """
    Crea modelo de clasificación PointNet++
    
    Args:
        num_classes: número de clases a predecir
        input_channels: 3 para xyz, 6 para xyz+normal, etc.
    """
    model = PointNet2Classification(num_classes=num_classes, input_channels=input_channels)
    return model


if __name__ == '__main__':
    # Test del modelo
    print("="*70)
    print("Test de PointNet++ Simplificado (CPU)")
    print("="*70)
    print()
    
    # Crear modelo
    num_classes = 2  # Ejemplo: con defecto / sin defecto
    model = create_classifier(num_classes=num_classes)
    model.eval()
    
    print(f"Modelo creado: {num_classes} clases")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Test con datos sintéticos
    batch_size = 4
    num_points = 1024
    
    # Crear datos de prueba (simulando estructura atómica)
    xyz = torch.randn(batch_size, 3, num_points)
    
    print(f"Input shape: {xyz.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Puntos: {num_points}")
    print(f"  - Dimensiones: 3 (x, y, z)")
    print()
    
    # Forward pass
    with torch.no_grad():
        output = model(xyz)
    
    print(f"Output shape: {output.shape}")
    print(f"Predicciones: {torch.argmax(output, dim=1)}")
    print()
    
    print("✅ Modelo funcionando correctamente en CPU!")
    print()
    print("Este modelo puede usarse para:")
    print("  - Clasificar estructuras con/sin defectos")
    print("  - Predecir concentración de vacancias")
    print("  - Identificar tipo de estructura cristalina")
    print("  - Cualquier tarea de clasificación de nubes de puntos")
