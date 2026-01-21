#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar la instalación
==============================================

Verifica que:
- PyTorch está instalado correctamente
- CUDA está disponible (si hay GPU)
- El modelo DGCNN se puede instanciar
- Las operaciones básicas funcionan

Uso:
    python test_installation.py
"""

import sys
from pathlib import Path

def test_pytorch():
    """Test PyTorch installation"""
    print("=" * 70)
    print("🔍 Verificando PyTorch...")
    print("-" * 70)

    try:
        import torch
        print(f"✅ PyTorch instalado: {torch.__version__}")

        # CUDA
        cuda_available = torch.cuda.is_available()
        print(f"{'✅' if cuda_available else '⚠️ '} CUDA disponible: {cuda_available}")

        if cuda_available:
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"           Memory: {props.total_memory / 1e9:.2f} GB")
        else:
            print("   ℹ️  Entrenamiento será en CPU (más lento)")

        return True

    except ImportError as e:
        print(f"❌ Error importando PyTorch: {e}")
        print("   Instalar con: pip install torch")
        return False


def test_dependencies():
    """Test other dependencies"""
    print("\n" + "=" * 70)
    print("🔍 Verificando dependencias...")
    print("-" * 70)

    deps = {
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard'
    }

    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✅ {name} instalado")
        except ImportError:
            print(f"❌ {name} no instalado")
            all_ok = False

    # OVITO (optional)
    try:
        import ovito
        print(f"✅ OVITO instalado (opcional)")
    except ImportError:
        print(f"⚠️  OVITO no instalado (opcional, para archivos dump)")

    return all_ok


def test_model():
    """Test DGCNN model"""
    print("\n" + "=" * 70)
    print("🔍 Verificando modelo DGCNN...")
    print("-" * 70)

    try:
        import torch
        from dgcnn_gpu.model import get_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Crear modelo
        model = get_model(task='classification', num_classes=4, k=20)
        model = model.to(device)
        print(f"✅ Modelo creado exitosamente")

        # Test forward pass
        batch_size = 4
        n_points = 256
        x = torch.randn(batch_size, n_points, 3).to(device)

        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out = model(x)
            else:
                out = model(x)

        print(f"✅ Forward pass exitoso")
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {out.shape}")

        # Parámetros
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parámetros: {n_params:,}")

        return True

    except Exception as e:
        print(f"❌ Error en modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_utils():
    """Test data utilities"""
    print("\n" + "=" * 70)
    print("🔍 Verificando utilidades de datos...")
    print("-" * 70)

    try:
        from utils.data_utils import normalize_points, resample_points, augment_points
        import numpy as np

        # Test normalización
        points = np.random.randn(100, 3).astype(np.float32)
        points_norm = normalize_points(points.copy())
        print(f"✅ Normalización funciona")

        # Test resampleo
        points_resampled = resample_points(points, n_points=256)
        assert len(points_resampled) == 256
        print(f"✅ Resampleo funciona")

        # Test augmentation
        points_aug = augment_points(points.copy())
        print(f"✅ Augmentation funciona")

        return True

    except Exception as e:
        print(f"❌ Error en utilidades: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary"""
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)

    all_ok = all(results.values())

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    print("\n" + "=" * 70)
    if all_ok:
        print("✅ INSTALACIÓN CORRECTA - ¡Listo para entrenar!")
        print("\n💡 Siguiente paso:")
        print("   1. Preparar datos en data/train/ y data/val/")
        print("   2. Ejecutar: python dgcnn_gpu/train.py --help")
    else:
        print("⚠️  ALGUNOS TESTS FALLARON")
        print("\n💡 Revisar:")
        print("   - pip install -r requirements.txt")
        print("   - Verificar instalación de PyTorch con CUDA")
    print("=" * 70)

    return all_ok


def main():
    """Run all tests"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "DGCNN-MD GPU - TEST DE INSTALACIÓN" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")

    results = {
        'PyTorch': test_pytorch(),
        'Dependencias': test_dependencies(),
        'Modelo DGCNN': test_model(),
        'Utilidades': test_data_utils()
    }

    all_ok = print_summary(results)

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
