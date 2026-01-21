#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERADOR DE DATOS PARA POINTNET++ v2
- Normalización robusta
- Validación de valores
- Extensión .off garantizada
- Split train/test en mismo directorio
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
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# OVITO
from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier, InvertSelectionModifier, DeleteSelectedModifier

# Configuración
ATM_TOTAL = 16384
A0 = 3.532


def extract_surface_coordinates(dump_path):
    """Extrae coordenadas de superficie usando OVITO"""
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
        positions = np.array(data_surface.particles['Position'][...], dtype=np.float64)
        
        return positions, n_vacancies
        
    except Exception as e:
        logger.debug(f"Error extrayendo superficie de {Path(dump_path).name}: {e}")
        return None, None


def normalize_coordinates(positions):
    """
    Normaliza coordenadas para PointNet++ de forma robusta
    Resultado: valores en rango aproximado [-1, 1]
    """
    if len(positions) == 0:
        return None
    
    positions = np.array(positions, dtype=np.float64)
    
    # Verificar NaN/Inf en entrada
    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
        logger.debug("Coordenadas de entrada contienen NaN o Inf")
        return None
    
    # Centrar en origen
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    
    # Escalar por distancia máxima al centroide
    distances = np.linalg.norm(centered, axis=1)
    max_dist = np.max(distances)
    
    if max_dist < 1e-10:  # Todos los puntos en el mismo lugar
        logger.debug("Todos los puntos están en la misma posición")
        return None
    
    normalized = centered / max_dist
    
    # Verificar resultado
    if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
        logger.debug("Normalización produjo NaN o Inf")
        return None
    
    # Verificar rango (debería estar en [-1, 1] aproximadamente)
    if np.max(np.abs(normalized)) > 1.5:
        logger.debug(f"Valores fuera de rango después de normalizar: max={np.max(np.abs(normalized))}")
        return None
    
    return normalized.astype(np.float32)


def write_off_file(positions, output_path, n_sample=256):
    """
    Escribe archivo OFF con muestreo y validación
    """
    if positions is None or len(positions) == 0:
        return False
    
    # Samplear
    n_points = len(positions)
    if n_sample < n_points:
        # Submuestreo aleatorio
        indices = np.random.choice(n_points, n_sample, replace=False)
        sampled = positions[indices]
    else:
        # Upsample con reemplazo + pequeño ruido para evitar duplicados exactos
        indices = np.random.choice(n_points, n_sample, replace=True)
        sampled = positions[indices].copy()
        # Agregar ruido muy pequeño a puntos duplicados
        noise = np.random.normal(0, 0.001, sampled.shape).astype(np.float32)
        sampled += noise
    
    # Re-normalizar después del ruido para garantizar rango
    max_val = np.max(np.abs(sampled))
    if max_val > 1.0:
        sampled = sampled / max_val
    
    # Validación final
    if np.any(np.isnan(sampled)) or np.any(np.isinf(sampled)):
        return False
    
    if np.max(np.abs(sampled)) > 1.5:
        return False
    
    # Crear directorio
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Escribir OFF
    with open(output_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(sampled)} 0 0\n")
        for pos in sampled:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    
    return True


def get_class_name(n_vac):
    """
    Determina clase según número de vacancias (3 clases)
    
    - small: 1-5 vacancias (single + small cluster)
    - medium: 6-9 vacancias
    - large: 10+ vacancias
    """
    if n_vac <= 5:
        return "small"    # 1-5 vacancias
    elif n_vac <= 9:
        return "medium"   # 6-9 vacancias
    else:
        return "large"    # 10+ vacancias


def process_dumps_to_off(input_pattern, csv_file, output_dir="data_pointnet", 
                        n_sample=256):
    """
    Procesa dumps y genera archivos OFF para PointNet++
    """
    
    logger.info("\n" + "="*70)
    logger.info("🎯 GENERADOR DE DATOS PARA POINTNET++ v2")
    logger.info("="*70)
    
    # Cargar CSV
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
    logger.info("="*70)
    
    output_base = Path(output_dir)
    
    # Estadísticas
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'failed_extract': 0,
        'failed_normalize': 0,
        'failed_write': 0,
        'by_class': {}
    }
    
    # Crear mapeo de nombres
    file_map = {}
    for dump_path in files:
        filename = Path(dump_path).name
        stem = Path(dump_path).stem
        
        # Intentar varios matches
        for candidate in [filename, stem, f"{stem}.dump"]:
            if candidate in df.index:
                file_map[dump_path] = candidate
                break
    
    logger.info(f"✅ Archivos mapeados: {len(file_map)}/{len(files)}")
    
    if len(file_map) == 0:
        logger.error("❌ No se pudo mapear ningún archivo!")
        return None
    
    # Procesar cada dump
    for i, (dump_path, csv_name) in enumerate(file_map.items(), 1):
        
        if i % 100 == 0:
            logger.info(f"Procesando: {i}/{len(file_map)} (exitosos: {stats['success']})")
        
        stats['total'] += 1
        
        # Obtener n_vacancies
        try:
            n_vac = int(df.loc[csv_name, 'n_vacancies'])
        except:
            stats['failed'] += 1
            continue
        
        # 1. Extraer coordenadas
        positions, _ = extract_surface_coordinates(dump_path)
        
        if positions is None or len(positions) < 4:
            stats['failed'] += 1
            stats['failed_extract'] += 1
            continue
        
        # 2. Normalizar
        normalized = normalize_coordinates(positions)
        
        if normalized is None:
            stats['failed'] += 1
            stats['failed_normalize'] += 1
            continue
        
        # 3. Determinar clase y path
        class_name = get_class_name(n_vac)
        class_dir = output_base / class_name
        
        # Nombre de archivo: garantizar extensión .off
        base_name = Path(dump_path).stem  # Sin extensión
        output_file = class_dir / f"{base_name}.off"
        
        # 4. Escribir
        success = write_off_file(normalized, output_file, n_sample=n_sample)
        
        if success:
            stats['success'] += 1
            stats['by_class'][class_name] = stats['by_class'].get(class_name, 0) + 1
        else:
            stats['failed'] += 1
            stats['failed_write'] += 1
    
    # Reporte
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESAMIENTO COMPLETADO")
    logger.info("="*70)
    logger.info(f"Total procesados: {stats['total']}")
    logger.info(f"Exitosos: {stats['success']}")
    logger.info(f"Fallidos: {stats['failed']}")
    logger.info(f"  - Extracción: {stats['failed_extract']}")
    logger.info(f"  - Normalización: {stats['failed_normalize']}")
    logger.info(f"  - Escritura: {stats['failed_write']}")
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
    Divide datos en train/test DENTRO del mismo directorio
    
    Estructura final:
    data_dir/
      train/
        large_cluster/
        medium_cluster/
        ...
      test/
        large_cluster/
        ...
    """
    logger.info("\n" + "="*70)
    logger.info("🔀 DIVIDIENDO EN TRAIN/TEST")
    logger.info("="*70)
    
    data_path = Path(data_dir)
    train_path = data_path / "train"
    test_path = data_path / "test"
    
    # Obtener clases (excluyendo train/test si ya existen)
    classes = [d for d in data_path.iterdir() 
               if d.is_dir() and d.name not in ['train', 'test']]
    
    if not classes:
        logger.warning("⚠️  No se encontraron clases en el directorio")
        return
    
    logger.info(f"Clases encontradas: {len(classes)}")
    logger.info(f"Split: {int((1-test_split)*100)}% train, {int(test_split*100)}% test")
    logger.info("")
    
    total_train = 0
    total_test = 0
    
    for class_dir in classes:
        class_name = class_dir.name
        files = sorted(list(class_dir.glob('*.off')))
        
        if not files:
            logger.warning(f"⚠️  {class_name}: sin archivos .off")
            continue
        
        # Shuffle reproducible
        np.random.seed(42)
        indices = np.random.permutation(len(files))
        
        # Calcular split
        n_test = max(1, int(len(files) * test_split))
        n_train = len(files) - n_test
        
        # Crear directorios destino
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Mover archivos (no copiar, para ahorrar espacio)
        for idx in indices[:n_train]:
            src = files[idx]
            dst = train_path / class_name / src.name
            shutil.move(str(src), str(dst))
        
        for idx in indices[n_train:]:
            src = files[idx]
            dst = test_path / class_name / src.name
            shutil.move(str(src), str(dst))
        
        total_train += n_train
        total_test += n_test
        logger.info(f"{class_name:20s}: {n_train:4d} train, {n_test:4d} test (total: {len(files)})")
    
    # Eliminar carpetas de clase vacías
    for class_dir in classes:
        if class_dir.exists() and not any(class_dir.iterdir()):
            class_dir.rmdir()
    
    logger.info("")
    logger.info("="*70)
    logger.info("✅ DIVISIÓN COMPLETADA")
    logger.info("="*70)
    logger.info(f"Total train: {total_train}")
    logger.info(f"Total test:  {total_test}")
    logger.info(f"Total:       {total_train + total_test}")
    logger.info("")
    logger.info(f"📁 Datos guardados en:")
    logger.info(f"   Train: {train_path}/")
    logger.info(f"   Test:  {test_path}/")
    logger.info("="*70)
    logger.info("")
    logger.info("✅ Ahora podés entrenar con:")
    logger.info("")
    logger.info(f"python train_pointnet2.py \\")
    logger.info(f"  --data {data_dir} \\")
    logger.info(f"  --n_points 256 \\")
    logger.info(f"  --epochs 50 \\")
    logger.info(f"  --batch_size 8 \\")
    logger.info(f"  --save_dir modelo_pointnet")


def main():
    parser = argparse.ArgumentParser(
        description="Generador de datos para PointNet++ v2 (con validación robusta)"
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Patrón de archivos dump (ej: "db_octubre_fcc_ni/*")')
    parser.add_argument('-c', '--csv', required=True,
                       help='CSV con features y n_vacancies')
    parser.add_argument('-o', '--output', default='data_pointnet',
                       help='Directorio de salida')
    parser.add_argument('-n', '--n_sample', type=int, default=256,
                       help='Puntos a samplear por estructura (default: 256)')
    parser.add_argument('--split', action='store_true',
                       help='Dividir en train/test después de generar')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fracción para test (default: 0.2)')
    
    args = parser.parse_args()
    
    # Limpiar directorio de salida si existe
    output_path = Path(args.output)
    if output_path.exists():
        logger.info(f"🗑️  Limpiando directorio existente: {output_path}")
        shutil.rmtree(output_path)
    
    # Generar archivos OFF
    stats = process_dumps_to_off(
        input_pattern=args.input,
        csv_file=args.csv,
        output_dir=args.output,
        n_sample=args.n_sample
    )
    
    if stats is None or stats['success'] == 0:
        logger.error("❌ Error: no se generaron archivos")
        return
    
    # Dividir en train/test
    if args.split:
        split_train_test(args.output, test_split=args.test_split)
    
    logger.info("\n✅ PROCESO COMPLETADO")


if __name__ == "__main__":
    main()
