#!/usr/bin/env python3
"""
Script de verificación rápida de instalación
Verifica que todas las dependencias estén correctamente instaladas
"""

import sys

def check_import(module_name, package_name=None):
    """Verifica si un módulo puede importarse"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - FALTA (Error: {e})")
        return False

def check_version(module_name, package_name=None):
    """Verifica versión de un paquete"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'desconocida')
        print(f"   └─ Versión: {version}")
        return version
    except:
        return None

def main():
    print("="*70)
    print("Verificación de Instalación - PointNet++ para Estructuras Atómicas")
    print("="*70)
    print()
    
    # Verificar módulos base
    print("📦 Verificando dependencias base...")
    print()
    
    all_ok = True
    
    modules = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('h5py', 'h5py'),
        ('tqdm', 'tqdm'),
    ]
    
    for module, name in modules:
        ok = check_import(module, name)
        if ok and module in ['numpy', 'torch']:
            check_version(module, name)
        all_ok = all_ok and ok
    
    print()
    
    # Verificar PyTorch específicamente
    print("🔍 Verificando configuración de PyTorch...")
    print()
    
    try:
        import torch
        print(f"   PyTorch versión: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        print(f"   CPU threads: {torch.get_num_threads()}")
        
        # Test rápido
        x = torch.randn(2, 3, 10)
        print(f"   Test tensor: {x.shape} - OK")
        print("   ✅ PyTorch funcionando correctamente")
    except Exception as e:
        print(f"   ❌ Error con PyTorch: {e}")
        all_ok = False
    
    print()
    
    # Verificar PointNet++
    print("🎯 Verificando PointNet++...")
    print()
    
    try:
        sys.path.append('./Pointnet_Pointnet2_pytorch')
        from models.pointnet2_cls_ssg import get_model
        
        # Intentar crear modelo
        model = get_model(num_class=2, normal_channel=False)
        print("   ✅ PointNet++ modelo cargado correctamente")
        
        # Verificar forward pass
        import torch
        x = torch.randn(2, 3, 1024)  # batch=2, xyz=3, points=1024
        with torch.no_grad():
            pred, _ = model(x)
        print(f"   ✅ Forward pass OK: {pred.shape}")
        
    except ImportError as e:
        print(f"   ❌ PointNet++ no encontrado")
        print(f"      Error: {e}")
        print(f"      Ejecuta: git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error al cargar PointNet++: {e}")
        all_ok = False
    
    print()
    
    # Verificar scripts personalizados
    print("📜 Verificando scripts personalizados...")
    print()
    
    from pathlib import Path
    scripts = [
        'dump_to_off.py',
        'train_atomic_example.py',
        'install_pointnet2.sh',
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} - NO ENCONTRADO")
    
    print()
    print("="*70)
    
    if all_ok:
        print("🎉 ¡TODO INSTALADO CORRECTAMENTE!")
        print()
        print("Próximos pasos:")
        print("  1. Convierte tus archivos .dump a .off:")
        print("     python dump_to_off.py estructura.dump estructura.off --normalize")
        print()
        print("  2. Organiza tus datos en carpetas por clase")
        print()
        print("  3. Entrena tu modelo:")
        print("     cd Pointnet_Pointnet2_pytorch")
        print("     python train_classification.py --model pointnet2_cls_ssg")
    else:
        print("⚠️  HAY PROBLEMAS CON LA INSTALACIÓN")
        print()
        print("Soluciones:")
        print("  1. Activa el entorno virtual:")
        print("     source pointnet2_env/bin/activate")
        print()
        print("  2. Instala dependencias faltantes:")
        print("     pip install torch numpy scipy matplotlib h5py tqdm")
        print()
        print("  3. Clona PointNet++ si falta:")
        print("     git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git")
    
    print("="*70)
    print()

if __name__ == '__main__':
    main()
