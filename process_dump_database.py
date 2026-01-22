#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procesador de Base de Datos Dump para DGCNN
===========================================

Organiza archivos dump de LAMMPS en estructura para entrenamiento:
    data_dir/
        train/
            clase1/
                archivo1.dump
                archivo2.dump
            clase2/
                archivo3.dump
        val/
            clase1/
                archivo4.dump
            clase2/
                archivo5.dump

Uso:
    # Procesar base de datos con split automático
    python process_dump_database.py --input_dir db_octubre_fcc_ni/ --output_dir data/ --classes vacancy cluster perfect

    # Con split personalizado
    python process_dump_database.py --input_dir raw_dumps/ --output_dir data/ --classes vacancy cluster --train_split 0.8

    # Copiar archivos (en lugar de mover)
    python process_dump_database.py --input_dir dumps/ --output_dir data/ --classes vacancy --copy
"""

import argparse
import shutil
from pathlib import Path
import random
from typing import List, Dict
import json


class DumpDatabaseProcessor:
    """Procesa y organiza base de datos de archivos dump"""

    def __init__(self, input_dir: str, output_dir: str, classes: List[str],
                 train_split: float = 0.8, copy_files: bool = False, seed: int = 42):
        """
        Args:
            input_dir: Directorio con archivos dump
            output_dir: Directorio de salida para train/val
            classes: Lista de nombres de clases
            train_split: Proporción para entrenamiento (0-1)
            copy_files: Si True, copia archivos; si False, los mueve
            seed: Semilla para reproducibilidad
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.classes = classes
        self.train_split = train_split
        self.copy_files = copy_files
        self.seed = seed

        random.seed(seed)

        if not self.input_dir.exists():
            raise ValueError(f"Directorio de entrada no existe: {input_dir}")

    def detect_class_from_filename(self, filepath: Path) -> str:
        """
        Detecta la clase de un archivo basándose en su nombre

        Args:
            filepath: Ruta al archivo

        Returns:
            Nombre de la clase o None si no se detecta
        """
        filename = filepath.name.lower()

        # Buscar clase en el nombre del archivo
        for class_name in self.classes:
            if class_name.lower() in filename:
                return class_name

        return None

    def find_dump_files(self) -> Dict[str, List[Path]]:
        """
        Encuentra y clasifica archivos dump

        Returns:
            Diccionario {clase: [archivos]}
        """
        print("\n📂 Buscando archivos dump...")

        files_by_class = {c: [] for c in self.classes}
        unclassified = []

        # Buscar archivos dump
        for pattern in ['*.dump', '*']:
            for filepath in self.input_dir.rglob(pattern):
                if not filepath.is_file():
                    continue

                # Ignorar archivos ocultos
                if filepath.name.startswith('.'):
                    continue

                # Detectar clase
                detected_class = self.detect_class_from_filename(filepath)

                if detected_class:
                    files_by_class[detected_class].append(filepath)
                elif filepath.suffix == '.dump' or self._looks_like_dump(filepath):
                    unclassified.append(filepath)

        # Resumen
        print(f"\n✅ Archivos encontrados:")
        total = 0
        for class_name, files in files_by_class.items():
            count = len(files)
            total += count
            if count > 0:
                print(f"  {class_name}: {count} archivos")

        if unclassified:
            print(f"\n⚠️  {len(unclassified)} archivos sin clasificar (ignorados)")
            print("   Sugerencia: renombra los archivos incluyendo el nombre de la clase")
            print(f"   Ejemplo: estructura_vacancy_001.dump")

        if total == 0:
            raise ValueError("No se encontraron archivos dump clasificados")

        return files_by_class

    def _looks_like_dump(self, filepath: Path) -> bool:
        """
        Verifica si un archivo parece ser un dump de LAMMPS

        Args:
            filepath: Ruta al archivo

        Returns:
            True si parece un dump de LAMMPS
        """
        try:
            with open(filepath, 'r') as f:
                first_lines = ''.join([f.readline() for _ in range(10)])
                return 'ITEM: TIMESTEP' in first_lines or 'ITEM: ATOMS' in first_lines
        except:
            return False

    def split_files(self, files: List[Path]) -> tuple:
        """
        Divide archivos en train/val

        Args:
            files: Lista de archivos

        Returns:
            (train_files, val_files)
        """
        # Mezclar aleatoriamente
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)

        # Split
        n_train = int(len(files) * self.train_split)
        train_files = files_shuffled[:n_train]
        val_files = files_shuffled[n_train:]

        return train_files, val_files

    def organize_files(self):
        """Organiza archivos en estructura train/val"""
        print("\n🔄 Organizando archivos...")

        # Encontrar archivos
        files_by_class = self.find_dump_files()

        # Crear estructura de directorios
        for split in ['train', 'val']:
            for class_name in self.classes:
                split_dir = self.output_dir / split / class_name
                split_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada clase
        stats = {
            'train': {c: 0 for c in self.classes},
            'val': {c: 0 for c in self.classes}
        }

        for class_name, files in files_by_class.items():
            if len(files) == 0:
                continue

            # Split train/val
            train_files, val_files = self.split_files(files)

            # Copiar/mover archivos train
            for filepath in train_files:
                dest = self.output_dir / 'train' / class_name / filepath.name
                if self.copy_files:
                    shutil.copy2(filepath, dest)
                else:
                    shutil.move(str(filepath), str(dest))
                stats['train'][class_name] += 1

            # Copiar/mover archivos val
            for filepath in val_files:
                dest = self.output_dir / 'val' / class_name / filepath.name
                if self.copy_files:
                    shutil.copy2(filepath, dest)
                else:
                    shutil.move(str(filepath), str(dest))
                stats['val'][class_name] += 1

        # Resumen
        action = "copiados" if self.copy_files else "movidos"
        print(f"\n✅ Archivos {action}:")
        print(f"\n📊 Train:")
        for class_name, count in stats['train'].items():
            if count > 0:
                print(f"  {class_name}: {count} archivos")

        print(f"\n📊 Validation:")
        for class_name, count in stats['val'].items():
            if count > 0:
                print(f"  {class_name}: {count} archivos")

        # Guardar estadísticas
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n💾 Estadísticas guardadas en: {stats_file}")

        return stats

    def validate_structure(self):
        """Valida que la estructura de directorios sea correcta"""
        print("\n🔍 Validando estructura...")

        valid = True

        # Verificar train/val
        for split in ['train', 'val']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                print(f"❌ Falta directorio: {split_dir}")
                valid = False
                continue

            # Verificar clases
            for class_name in self.classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    print(f"⚠️  Directorio vacío: {class_dir}")
                    continue

                # Contar archivos
                files = list(class_dir.glob('*'))
                if len(files) == 0:
                    print(f"⚠️  No hay archivos en: {class_dir}")

        if valid:
            print("✅ Estructura válida")

        return valid


def main():
    parser = argparse.ArgumentParser(
        description='Procesa base de datos dump para entrenamiento DGCNN'
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directorio con archivos dump')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directorio de salida (creará train/ y val/)')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Nombres de clases (ej: vacancy cluster perfect)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proporción para entrenamiento (default: 0.8)')
    parser.add_argument('--copy', action='store_true',
                        help='Copiar archivos en lugar de moverlos')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria (default: 42)')

    args = parser.parse_args()

    # Validar argumentos
    if not 0 < args.train_split < 1:
        raise ValueError("train_split debe estar entre 0 y 1")

    # Procesar
    print("=" * 70)
    print("📁 PROCESADOR DE BASE DE DATOS DUMP")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Clases: {', '.join(args.classes)}")
    print(f"Split:  {args.train_split*100:.0f}% train / {(1-args.train_split)*100:.0f}% val")
    print(f"Modo:   {'Copiar' if args.copy else 'Mover'}")
    print("=" * 70)

    processor = DumpDatabaseProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        train_split=args.train_split,
        copy_files=args.copy,
        seed=args.seed
    )

    # Organizar archivos
    stats = processor.organize_files()

    # Validar
    processor.validate_structure()

    print("\n" + "=" * 70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 70)
    print("\n📖 Siguiente paso: entrenar el modelo")
    print(f"\nPython dgcnn_gpu/train.py --data_dir {args.output_dir} --classes {' '.join(args.classes)} --from_dump --mixed_precision")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
