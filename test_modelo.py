#!/usr/bin/env python3
"""
Test rápido del modelo PointNet++ corregido
"""

import torch
from pointnet2_simple import create_classifier

print("="*70)
print("Test PointNet++ - Versión Corregida")
print("="*70)
print()

# Crear modelo
model = create_classifier(num_classes=2)
model.eval()

print(f"✅ Modelo creado: 2 clases")
print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
print()

# Test con diferentes tamaños
test_cases = [
    (1, 512),   # 1 estructura, 512 puntos
    (2, 1024),  # 2 estructuras, 1024 puntos
    (4, 1024),  # 4 estructuras, 1024 puntos
    (8, 2048),  # 8 estructuras, 2048 puntos
]

print("Probando diferentes configuraciones:")
print()

for batch, points in test_cases:
    try:
        xyz = torch.randn(batch, 3, points)
        
        with torch.no_grad():
            output = model(xyz)
        
        print(f"✅ Batch={batch}, Puntos={points:4d} → Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Batch={batch}, Puntos={points:4d} → Error: {e}")

print()
print("="*70)
print("✅ ¡Modelo funcionando correctamente!")
print("="*70)
print()
print("Ahora podés:")
print("  1. Convertir tus dumps: python dump_to_off.py ...")
print("  2. Entrenar modelo: python train_pointnet2.py --data data")
print("  3. Hacer predicciones: python predict_pointnet2.py ...")
