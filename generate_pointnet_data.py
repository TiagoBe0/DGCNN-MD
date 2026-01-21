#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERADOR DE DATOS PARA POINTNET++
Integrado con opentopologyc_extractor.py

Toma los mismos dumps procesados y genera archivos .off
para entrenar PointNet++ con coordenadas xyz directamente
"""

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# OVITO
from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier, InvertSelectionModifier, DeleteSelectedModifier

# Configuración (misma que opentopologyc_extractor.py)
ATM_TOTAL = 16384
A0 = 3.532


def extract_surface_coordinates(dump_path):
    """
    Extrae coordenadas de superficie usando OVITO
    (Mismo proceso que opentopologyc_extractor.py)
    """
    try:
        pipeline = import_file(dump_path)
        
        # Alpha shape (surface detection)
        pipeline.modifiers.append(ConstructSurfaceModifier(
            radius=2.0,
            select_surface_particles=True,
            smoothing_level=12
        ))
        
        # Contar vacancias
        data_full = pipeline.compute()
        n_vacancies = ATM_TOTAL - data_full.particles.count
        
        # Invertir selección y borrar bulk
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        
        # Obtener solo superficie
        data_surface = pipeline.compute()
        positions = data_surface.particles['Position'][...]
        
        return positions, n_vacancies
        
    except Exception as e:
        logger.error(f"Error extrayendo superficie de {Path(dump_path).name}: {e}")
        return None, None


def normalize_coordinates(positions):
    """
    Normaliza coordenadas para PointNet++
    - Centra en origen
    - Escala por distancia máxima
    """
    if len(positions) == 0:
        return positions
    
    # Centrar en origen
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    
    # Escalar por distancia máxima
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        normalized = centered / max_dist
    else:
        normalized = centered
    
    return normalized


def write_off_file(positions, output_path, n_sample=None):
    """
    Escribe archivo OFF para PointNet++
    
    Args:
        positions: coordenadas (N, 3)
        output_path: ruta de salida
        n_sample: número de puntos a samplear (None = todos)
    """
    if len(positions) == 0:
        logger.warning(f"No hay puntos para escribir en {output_path}")
        return False
    
    # Samplear si se especificó
    if n_sample and n_sample < len(positions):
        indices = np.random.choice(len(positions), n_sample, replace=False)
        sampled = positions[indices]
    elif n_sample and n_sample > len(positions):
        # Upsample con reemplazo
        indices = np.random.choice(len(positions), n_sample, replace=True)
        sampled = positions[indices]
    else:
        sampled = positions
    
    # Crear directorio si no existe
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Escribir OFF
    with open(output_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(sampled)} 0 0\n")
        
        for pos in sampled:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    
    return True


def process_dumps_to_off(input_pattern, csv_file, output_dir="data_pointnet", 
                        n_sample=256, normalize=True):
    """
    Procesa dumps y genera archivos OFF para PointNet++
    
    Args:
        input_pattern: patrón de archivos dump (ej: "databases/db_integrate/*")
        csv_file: CSV con features (para obtener n_vacancies)
        output_dir: directorio de salida
        n_sample: puntos a samplear por estructura
        normalize: si normalizar coordenadas
    """
    
    logger.info("\n" + "="*70)
    logger.info("🎯 GENERADOR DE DATOS PARA POINTNET++")
    logger.info("="*70)
    
    # Cargar CSV con n_vacancies
    try:
        df = pd.read_csv(csv_file, index_col='file')
        logger.info(f"✅ CSV cargado: {len(df)} muestras")
    except Exception as e:
        logger.error(f"❌ Error cargando CSV: {e}")
        return None
    
    # Obtener archivos dump
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        logger.error(f"❌ No se encontraron archivos: {input_pattern}")
        return None
    
    logger.info(f"📁 Archivos dump encontrados: {len(files)}")
    logger.info(f"🎲 Puntos por muestra: {n_sample}")
    logger.info(f"📐 Normalizar: {'Sí' if normalize else 'No'}")
    logger.info("="*70)
    
    # Crear directorios por clase (según n_vacancies)
    output_base = Path(output_dir)
    
    # Estadísticas
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'by_class': {}
    }
    
    # Debugging: mostrar primeros archivos
    logger.info(f"\n📋 Primeros 5 archivos dump encontrados:")
    for f in files[:5]:
        logger.info(f"   {Path(f).name}")
    
    logger.info(f"\n📋 Primeros 5 nombres en CSV:")
    for name in df.index[:5]:
        logger.info(f"   {name}")
    
    logger.info("")
    
    # Crear mapeo de nombres (flexible)
    # Intentar diferentes variantes de matching
    file_map = {}  # dump_path -> csv_name
    
    for dump_path in files:
        filename = Path(dump_path).name
        
        # Intentar match exacto
        if filename in df.index:
            file_map[dump_path] = filename
            continue
        
        # Intentar sin extensión y con .dump
        base_name = Path(dump_path).stem
        if base_name in df.index:
            file_map[dump_path] = base_name
            continue
        if f"{base_name}.dump" in df.index:
            file_map[dump_path] = f"{base_name}.dump"
            continue
    
    logger.info(f"✅ Archivos mapeados: {len(file_map)}/{len(files)}")
    
    if len(file_map) == 0:
        logger.error("❌ No se pudo mapear ningún archivo!")
        logger.error("Los nombres en el directorio no coinciden con los del CSV")
        logger.info("\nPrimeros nombres en directorio:")
        for f in files[:10]:
            logger.info(f"  {Path(f).name}")
        logger.info("\nPrimeros nombres en CSV:")
        for name in df.index[:10]:
            logger.info(f"  {name}")
        return None
    
    # Procesar cada dump
    for i, (dump_path, csv_name) in enumerate(file_map.items(), 1):
        
        if i % 100 == 0:
            logger.info(f"Procesando: {i}/{len(file_map)}")
        
        stats['total'] += 1
        n_vac = int(df.loc[csv_name, 'n_vacancies'])
        
        # Extraer coordenadas de superficie
        positions, _ = extract_surface_coordinates(dump_path)
        
        if positions is None or len(positions) < 4:
            stats['failed'] += 1
            continue
        
        # Normalizar si se pidió
        if normalize:
            positions = normalize_coordinates(positions)
        
        # Determinar carpeta de salida según n_vacancies
        # Estrategia: agrupar en clases
        # Opción 1: Single (1-2) vs Small (3-5) vs Medium (6-9) vs Large (10+)
        if n_vac <= 2:
            class_name = "single_vacancy"  # 1-2 vacancias
        elif n_vac <= 5:
            class_name = "small_cluster"   # 3-5 vacancias
        elif n_vac <= 9:
            class_name = "medium_cluster"  # 6-9 vacancias
        else:
            class_name = "large_cluster"   # 10+ vacancias
        
        # Otra opción: por número exacto (si tenés suficientes muestras)
        # class_name = f"vac_{n_vac:02d}"
        
        # Crear directorio
        class_dir = output_base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Escribir OFF
        dump_filename = Path(dump_path).name
        output_file = class_dir / dump_filename.replace('.dump', '.off')
        success = write_off_file(positions, output_file, n_sample=n_sample)
        
        if success:
            stats['success'] += 1
            stats['by_class'][class_name] = stats['by_class'].get(class_name, 0) + 1
    
    # Reporte final
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESAMIENTO COMPLETADO")
    logger.info("="*70)
    logger.info(f"Total dumps procesados: {stats['total']}")
    logger.info(f"Exitosos: {stats['success']}")
    logger.info(f"Fallidos: {stats['failed']}")
    logger.info("")
    logger.info("📊 Distribución por clase:")
    for class_name, count in sorted(stats['by_class'].items()):
        logger.info(f"   {class_name:20s}: {count:4d} muestras")
    logger.info("")
    logger.info(f"📁 Datos guardados en: {output_base}/")
    logger.info("="*70)
    
    return stats


def split_train_test(data_dir, test_split=0.2):
    """
    Divide datos en train/test manteniendo balance de clases
    
    Args:
        data_dir: directorio con clases
        test_split: fracción para test (default 0.2 = 20%)
    """
    logger.info("\n" + "="*70)
    logger.info("🔀 DIVIDIENDO EN TRAIN/TEST")
    logger.info("="*70)
    
    data_path = Path(data_dir)
    train_path = data_path.parent / f"{data_path.name}_split" / "train"
    test_path = data_path.parent / f"{data_path.name}_split" / "test"
    
    # Obtener todas las clases
    classes = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not classes:
        logger.error("❌ No se encontraron clases en el directorio")
        return
    
    logger.info(f"Clases encontradas: {len(classes)}")
    logger.info(f"Split: {int((1-test_split)*100)}% train, {int(test_split*100)}% test")
    logger.info("")
    
    for class_dir in classes:
        class_name = class_dir.name
        files = sorted(list(class_dir.glob('*.off')))
        
        if not files:
            continue
        
        # Calcular split
        n_test = max(1, int(len(files) * test_split))
        n_train = len(files) - n_test
        
        # Shuffle reproducible
        np.random.seed(42)
        indices = np.random.permutation(len(files))
        
        # Crear directorios
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copiar archivos
        import shutil
        
        for idx in indices[:n_train]:
            src = files[idx]
            dst = train_path / class_name / src.name
            shutil.copy2(src, dst)
        
        for idx in indices[n_train:]:
            src = files[idx]
            dst = test_path / class_name / src.name
            shutil.copy2(src, dst)
        
        logger.info(f"{class_name:20s}: {n_train:4d} train, {n_test:4d} test")
    
    logger.info("")
    logger.info(f"✅ Datos divididos guardados en:")
    logger.info(f"   Train: {train_path}/")
    logger.info(f"   Test:  {test_path}/")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Generador de datos para PointNet++ desde dumps OVITO"
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Patrón de archivos dump (ej: "databases/db_integrate/*")')
    parser.add_argument('-c', '--csv', required=True,
                       help='CSV con features y n_vacancies')
    parser.add_argument('-o', '--output', default='data_pointnet',
                       help='Directorio de salida')
    parser.add_argument('-n', '--n_sample', type=int, default=256,
                       help='Puntos a samplear por estructura (default: 256)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='No normalizar coordenadas')
    parser.add_argument('--split', action='store_true',
                       help='Dividir en train/test después de generar')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fracción para test (default: 0.2)')
    
    args = parser.parse_args()
    
    # Generar archivos OFF
    stats = process_dumps_to_off(
        input_pattern=args.input,
        csv_file=args.csv,
        output_dir=args.output,
        n_sample=args.n_sample,
        normalize=not args.no_normalize
    )
    
    if stats is None:
        logger.error("❌ Error en procesamiento")
        return
    
    # Dividir en train/test si se pidió
    if args.split and stats and stats['success'] > 0:
        split_train_test(args.output, test_split=args.test_split)
    elif args.split:
        logger.warning("⚠️  No hay datos para dividir en train/test")
    
    logger.info("\n✅ PROCESO COMPLETADO")


if __name__ == "__main__":
    main()

    """
    EJEMPLOS DE USO:
    
    # Generar archivos OFF con 256 puntos
    python generate_pointnet_data.py \
        -i "databases/db_integrate/*" \
        -c dataset_geometry_preserved.csv \
        -o data_pointnet \
        -n 256
    
    # Generar y dividir en train/test automáticamente
    python generate_pointnet_data.py \
        -i "databases/db_integrate/*" \
        -c dataset_geometry_preserved.csv \
        -o data_pointnet \
        -n 256 \
        --split \
        --test-split 0.2
    
    # Con 512 puntos (para clusters más grandes)
    python generate_pointnet_data.py \
        -i "databases/db_integrate/*" \
        -c dataset_geometry_preserved.csv \
        -o data_pointnet \
        -n 512 \
        --split
    """
