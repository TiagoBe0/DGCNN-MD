#!/usr/bin/env python3
"""
PointNet++ optimizado para clusters pequeños de vacancias
Arquitectura ajustada para 50-500 átomos superficiales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importar funciones base
import sys
sys.path.append('.')
from pointnet2_simple import (
    PointNetSetAbstraction, 
    create_classifier as create_base_classifier
)


class PointNet2VacancyClassifier(nn.Module):
    """
    PointNet++ optimizado para clasificar clusters de vacancias
    
    Diferencias vs modelo estándar:
    - Menos downsampling (para preservar detalles en clusters pequeños)
    - Radios más pequeños (estructura local más importante)
    - Menos parámetros (evitar overfitting con pocos datos)
    """
    def __init__(self, num_classes=2, input_channels=3):
        super(PointNet2VacancyClassifier, self).__init__()
        
        # Set Abstraction con parámetros ajustados para clusters pequeños
        # Capa 1: 256 → 128 puntos (en lugar de 512 → 128)
        self.sa1 = PointNetSetAbstraction(
            npoint=128, radius=0.15, nsample=16,  # Radio pequeño, pocos vecinos
            in_channel=input_channels, mlp=[32, 32, 64]
        )
        
        # Capa 2: 128 → 32 puntos
        self.sa2 = PointNetSetAbstraction(
            npoint=32, radius=0.3, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128]
        )
        
        # Capa 3: Global (todos los puntos)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=128 + 3, mlp=[128, 256, 512], group_all=True
        )
        
        # Clasificación final (más simple que el estándar)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)  # Menos dropout
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, xyz):
        B, _, _ = xyz.shape
        
        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Clasificación
        x = l3_points.reshape(B, 512)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x


def create_vacancy_classifier(num_classes=2, input_channels=3):
    """
    Crea modelo optimizado para clusters de vacancias
    
    Args:
        num_classes: número de clases (ej: 2 para single/cluster, 
                     4 para single/2-vacancy/3-vacancy/cluster)
        input_channels: 3 para xyz
    """
    model = PointNet2VacancyClassifier(num_classes=num_classes, input_channels=input_channels)
    return model


if __name__ == '__main__':
    print("="*70)
    print("PointNet++ para Clasificación de Clusters de Vacancias")
    print("="*70)
    print()
    
    # Crear modelo
    num_classes = 2  # single vacancy vs cluster
    model = create_vacancy_classifier(num_classes=num_classes)
    model.eval()
    
    print(f"Modelo creado: {num_classes} clases")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Test con diferentes tamaños (típicos de clusters pequeños)
    test_cases = [
        (4, 64),   # Clusters muy pequeños
        (4, 128),  # Single vacancies típicas
        (4, 256),  # Clusters medianos
        (4, 512),  # Clusters grandes
    ]
    
    print("Probando con tamaños típicos de clusters de vacancias:")
    print()
    
    for batch, points in test_cases:
        xyz = torch.randn(batch, 3, points)
        
        with torch.no_grad():
            output = model(xyz)
        
        print(f"✅ Batch={batch}, Puntos={points:4d} → Output shape: {output.shape}")
    
    print()
    print("="*70)
    print("Comparación de modelos:")
    print("="*70)
    
    # Comparar con modelo estándar
    model_standard = create_base_classifier(num_classes=2)
    
    params_vacancy = sum(p.numel() for p in model.parameters())
    params_standard = sum(p.numel() for p in model_standard.parameters())
    
    print(f"Modelo estándar:  {params_standard:,} parámetros")
    print(f"Modelo vacancias: {params_vacancy:,} parámetros")
    print(f"Reducción:        {params_standard - params_vacancy:,} parámetros ({100*(params_standard-params_vacancy)/params_standard:.1f}%)")
    print()
    print("Ventajas del modelo para vacancias:")
    print("  - Menos parámetros → menos propenso a overfitting")
    print("  - Radios pequeños → captura detalles locales finos")
    print("  - Menos downsampling → preserva geometría de clusters pequeños")
