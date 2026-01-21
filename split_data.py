#!/usr/bin/env python3
"""
Script simple para dividir data_pointnet en train/test
"""

import shutil
from pathlib import Path
import numpy as np

def split_data(input_dir="data_pointnet", output_dir="data_pointnet", test_split=0.2):
    """Divide datos en train/test"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Crear estructura de directorios
    train_path = output_path / "train"
    test_path = output_path / "test"
    
    print("="*70)
    print("🔀 DIVIDIENDO DATOS EN TRAIN/TEST")
    print("="*70)
    print()
    
    # Obtener todas las clases
    classes = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not classes:
        print("❌ No se encontraron clases en", input_dir)
        return
    
    print(f"Clases encontradas: {len(classes)}")
    print(f"Split: {int((1-test_split)*100)}% train, {int(test_split*100)}% test")
    print()
    
    np.random.seed(42)  # Reproducible
    
    total_train = 0
    total_test = 0
    
    for class_dir in classes:
        class_name = class_dir.name
        
        # Obtener todos los archivos .off
        files = sorted(list(class_dir.glob('*.off')))
        
        if not files:
            print(f"⚠️  {class_name}: sin archivos .off")
            continue
        
        # Calcular split
        n_test = max(1, int(len(files) * test_split))
        n_train = len(files) - n_test
        
        # Shuffle y dividir
        indices = np.random.permutation(len(files))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Crear directorios
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copiar archivos a train
        for idx in train_indices:
            src = files[idx]
            dst = train_path / class_name / src.name
            shutil.copy2(src, dst)
        
        # Copiar archivos a test
        for idx in test_indices:
            src = files[idx]
            dst = test_path / class_name / src.name
            shutil.copy2(src, dst)
        
        total_train += n_train
        total_test += n_test
        
        print(f"{class_name:20s}: {n_train:4d} train, {n_test:4d} test  (total: {len(files)})")
    
    print()
    print("="*70)
    print("✅ DIVISIÓN COMPLETADA")
    print("="*70)
    print(f"Total train: {total_train}")
    print(f"Total test:  {total_test}")
    print(f"Total:       {total_train + total_test}")
    print()
    print(f"📁 Datos guardados en:")
    print(f"   Train: {train_path}/")
    print(f"   Test:  {test_path}/")
    print("="*70)


if __name__ == '__main__':
    split_data()
    print()
    print("✅ Ahora podés entrenar con:")
    print()
    print("python train_pointnet2.py \\")
    print("  --data data_pointnet \\")
    print("  --n_points 256 \\")
    print("  --epochs 10 \\")
    print("  --batch_size 8 \\")
    print("  --save_dir modelo_test")
