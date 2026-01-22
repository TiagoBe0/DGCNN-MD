#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificación de Setup para Entrenamiento con Dumps
==================================================

Verifica que todo esté listo para entrenar:
- PyTorch y CUDA
- OVITO
- Estructura de datos
- GPU disponible
- Archivos dump válidos

Uso:
    python verify_setup.py --data_dir data/ --classes vacancy cluster
"""

import argparse
import sys
from pathlib import Path
import subprocess


def check_pytorch():
    """Verifica instalación de PyTorch"""
    print("\n🔍 Verificando PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponible")
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"   ⚠️  CUDA no disponible (entrenamiento será lento en CPU)")

        return True
    except ImportError:
        print("   ❌ PyTorch no instalado")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_ovito():
    """Verifica instalación de OVITO"""
    print("\n🔍 Verificando OVITO...")
    try:
        from ovito.io import import_file
        from ovito.modifiers import ConstructSurfaceModifier
        print("   ✅ OVITO instalado correctamente")
        return True
    except ImportError as e:
        print("   ❌ OVITO no instalado o incompleto")
        print("      pip install ovito")
        return False


def check_dependencies():
    """Verifica otras dependencias"""
    print("\n🔍 Verificando dependencias...")

    dependencies = [
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('tensorboard', 'TensorBoard'),
    ]

    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} no instalado")
            all_ok = False

    if not all_ok:
        print("\n   Instalar todas las dependencias:")
        print("   pip install -r requirements.txt")

    return all_ok


def check_data_structure(data_dir, classes):
    """Verifica estructura de datos"""
    print(f"\n🔍 Verificando estructura de datos en {data_dir}...")

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"   ❌ Directorio no existe: {data_dir}")
        print(f"\n   Primero procesa tu base de datos:")
        print(f"   python process_dump_database.py --input_dir <tu_db> --output_dir {data_dir} --classes {' '.join(classes)}")
        return False

    # Verificar train/val
    all_ok = True
    for split in ['train', 'val']:
        split_path = data_path / split
        if not split_path.exists():
            print(f"   ❌ Falta directorio: {split}/")
            all_ok = False
            continue

        print(f"   ✅ {split}/")

        # Verificar clases
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"      ❌ Falta clase: {split}/{class_name}/")
                all_ok = False
                continue

            # Contar archivos
            files = list(class_path.glob('*'))
            n_files = len([f for f in files if f.is_file()])

            if n_files == 0:
                print(f"      ⚠️  {split}/{class_name}/: 0 archivos")
            else:
                print(f"      ✅ {split}/{class_name}/: {n_files} archivos")

    if not all_ok:
        print(f"\n   Estructura esperada:")
        print(f"   {data_dir}/")
        print(f"     train/")
        for c in classes:
            print(f"       {c}/")
        print(f"     val/")
        for c in classes:
            print(f"       {c}/")

    return all_ok


def check_dump_file(filepath):
    """Verifica si un archivo es un dump válido de LAMMPS"""
    try:
        with open(filepath, 'r') as f:
            content = f.read(500)  # Leer primeros 500 caracteres

            required = ['ITEM: TIMESTEP', 'ITEM: ATOMS']
            has_all = all(req in content for req in required)

            return has_all
    except:
        return False


def check_sample_dumps(data_dir, classes):
    """Verifica algunos archivos dump de muestra"""
    print(f"\n🔍 Verificando archivos dump de muestra...")

    data_path = Path(data_dir)

    # Tomar 1 archivo de cada clase en train
    for class_name in classes:
        class_path = data_path / 'train' / class_name
        if not class_path.exists():
            continue

        files = list(class_path.glob('*'))[:1]  # Solo 1 archivo

        for filepath in files:
            if filepath.is_file():
                is_valid = check_dump_file(filepath)
                status = "✅" if is_valid else "❌"
                print(f"   {status} {class_name}/{filepath.name}")

                if not is_valid:
                    print(f"      Archivo no parece ser un dump válido de LAMMPS")


def check_gpu_memory(batch_size=32, n_points=256):
    """Estima uso de memoria GPU"""
    print(f"\n🔍 Estimando uso de memoria GPU...")

    try:
        import torch

        if not torch.cuda.is_available():
            print("   ⚠️  GPU no disponible, skipping")
            return True

        # Memoria disponible
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Estimación burda: ~50MB por muestra con 256 puntos
        estimated_usage = (batch_size * n_points * 50e-6)

        print(f"   VRAM total: {total_memory:.1f} GB")
        print(f"   Batch size: {batch_size}")
        print(f"   Puntos: {n_points}")
        print(f"   Uso estimado: ~{estimated_usage:.1f} GB")

        if estimated_usage > total_memory * 0.8:
            print(f"   ⚠️  Puede quedarse sin memoria")
            print(f"   Sugerencia: reducir batch_size a {int(batch_size * 0.5)} o menos")
        else:
            print(f"   ✅ Debería funcionar correctamente")

        return True

    except:
        return True


def print_summary(checks):
    """Imprime resumen de verificaciones"""
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)

    all_passed = all(checks.values())

    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    print("=" * 70)

    if all_passed:
        print("✅ TODO LISTO PARA ENTRENAR")
        print("\n🚀 Siguiente paso:")
        print("   python train_dump_example.py --data_dir <tu_dir> --classes <tus_clases>")
    else:
        print("❌ FALTAN CONFIGURACIONES")
        print("\n📖 Revisa los errores arriba y corrígelos")

    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Verificar setup para entrenamiento')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directorio de datos')
    parser.add_argument('--classes', nargs='+', default=['vacancy'],
                        help='Clases a verificar')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size para estimación de memoria')
    parser.add_argument('--n_points', type=int, default=256,
                        help='Número de puntos para estimación')

    args = parser.parse_args()

    print("=" * 70)
    print("🔧 VERIFICACIÓN DE SETUP DGCNN")
    print("=" * 70)

    # Ejecutar verificaciones
    checks = {
        'PyTorch': check_pytorch(),
        'OVITO': check_ovito(),
        'Dependencias': check_dependencies(),
        'Estructura de datos': check_data_structure(args.data_dir, args.classes),
    }

    # Verificaciones adicionales solo si lo básico está ok
    if checks['Estructura de datos']:
        check_sample_dumps(args.data_dir, args.classes)

    if checks['PyTorch']:
        check_gpu_memory(args.batch_size, args.n_points)

    # Resumen
    all_ok = print_summary(checks)

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
