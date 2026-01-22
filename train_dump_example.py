#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de Entrenamiento con Base de Datos Dump
===============================================

Script de ejemplo para entrenar DGCNN con archivos dump procesados

Uso:
    # Entrenamiento básico
    python train_dump_example.py

    # Con GPU y mixed precision
    python train_dump_example.py --mixed_precision

    # Personalizar parámetros
    python train_dump_example.py --data_dir data/ --classes vacancy cluster perfect --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Añadir directorios al path
sys.path.append(str(Path(__file__).parent))

from dgcnn_gpu.train import Trainer


def main():
    parser = argparse.ArgumentParser(description='Entrenar DGCNN con archivos dump')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directorio con datos (debe tener train/ y val/)')
    parser.add_argument('--classes', nargs='+', default=['vacancy'],
                        help='Nombres de clases (default: vacancy)')
    parser.add_argument('--n_points', type=int, default=256,
                        help='Número de puntos de superficie a extraer')

    # Parámetros de superficie OVITO
    parser.add_argument('--radius', type=float, default=2.0,
                        help='Radio para construcción de superficie (default: 2.0)')
    parser.add_argument('--smoothing', type=int, default=12,
                        help='Nivel de suavizado de superficie (default: 12)')

    # Model
    parser.add_argument('--k', type=int, default=20,
                        help='Número de vecinos en k-NN')
    parser.add_argument('--emb_dims', type=int, default=1024,
                        help='Dimensión del embedding')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamaño del batch (reducir si hay problemas de memoria)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='LR scheduler')

    # GPU Optimization
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Usar mixed precision (AMP) - recomendado para GPU')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Usar múltiples GPUs si están disponibles')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Workers para DataLoader')
    parser.add_argument('--cache_data', action='store_true',
                        help='Cachear datos en memoria (solo para datasets pequeños)')

    # Other
    parser.add_argument('--output_dir', type=str, default='outputs/dgcnn_dump',
                        help='Directorio de salida')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # Siempre usar archivos dump
    parser.add_argument('--from_dump', action='store_true', default=True,
                        help='Cargar desde archivos dump (siempre True en este script)')

    args = parser.parse_args()

    # Verificar que existe el directorio de datos
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Directorio de datos no existe: {data_dir}")
        print("\n📖 Primero debes procesar tu base de datos:")
        print(f"   python process_dump_database.py --input_dir <tu_db> --output_dir {data_dir} --classes {' '.join(args.classes)}")
        sys.exit(1)

    # Verificar estructura
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ Error: Falta estructura train/val en {data_dir}")
        print("\n📖 Estructura esperada:")
        print(f"   {data_dir}/")
        print("     train/")
        print("       clase1/")
        print("       clase2/")
        print("     val/")
        print("       clase1/")
        print("       clase2/")
        sys.exit(1)

    # Información
    print("=" * 70)
    print("🔷 ENTRENAMIENTO DGCNN CON DUMPS")
    print("=" * 70)
    print(f"📁 Data dir: {args.data_dir}")
    print(f"🏷️  Clases: {', '.join(args.classes)}")
    print(f"🔢 Puntos de superficie: {args.n_points}")
    print(f"⚙️  Parámetros OVITO:")
    print(f"   - Radio: {args.radius}")
    print(f"   - Suavizado: {args.smoothing}")
    print(f"🚀 Batch size: {args.batch_size}")
    print(f"📊 Épocas: {args.epochs}")
    print(f"💾 Output: {args.output_dir}")

    if args.cache_data:
        print("⚠️  Cache activado - asegúrate de tener suficiente RAM")

    print("=" * 70)

    # Confirmar antes de empezar
    response = input("\n¿Iniciar entrenamiento? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Cancelado")
        sys.exit(0)

    # Entrenar
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
