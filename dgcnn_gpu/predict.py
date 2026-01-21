#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGCNN Prediction Script - GPU Optimized
========================================

Script de predicción optimizado para GPU con:
- Batch inference para mayor velocidad
- Mixed precision
- Exportación de resultados

Uso:
    # Predecir un archivo
    python predict.py --model outputs/dgcnn_gpu --input archivo.off

    # Predecir directorio con batch inference
    python predict.py --model outputs/dgcnn_gpu --input data/ --batch_size 64

    # Desde archivos dump
    python predict.py --model outputs/dgcnn_gpu --input data/ --from_dump

    # Exportar resultados
    python predict.py --model outputs/dgcnn_gpu --input data/ --output results.csv
"""

import argparse
import sys
from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Añadir directorios al path
sys.path.append(str(Path(__file__).parent.parent))

from dgcnn_gpu.model import get_model
from utils.data_utils import load_off, load_dump_surface, load_config


class Predictor:
    """Clase para predicción con DGCNN"""

    def __init__(self, model_dir, device=None, mixed_precision=True):
        """
        Args:
            model_dir: directorio del modelo entrenado
            device: dispositivo (cuda/cpu)
            mixed_precision: usar mixed precision
        """
        self.model_dir = Path(model_dir)
        self.mixed_precision = mixed_precision

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Cargar config
        config_path = self.model_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config no encontrado: {config_path}")

        self.config = load_config(config_path)
        self.classes = self.config['classes']
        self.n_points = self.config.get('n_points', 256)
        self.k = self.config.get('k', 20)
        self.emb_dims = self.config.get('emb_dims', 1024)
        self.dropout = self.config.get('dropout', 0.5)

        # Cargar modelo
        self._load_model()

        print("=" * 70)
        print("🔷 DGCNN Predictor - GPU Optimized")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Classes: {self.classes}")
        print(f"Points: {self.n_points}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print("-" * 70)

    def _load_model(self):
        """Carga modelo entrenado"""
        # Buscar checkpoint
        checkpoint_path = self.model_dir / 'best_model.pth'
        if not checkpoint_path.exists():
            checkpoint_path = self.model_dir / 'last_model.pth'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No se encontró checkpoint en {self.model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Crear modelo
        self.model = get_model(
            task='classification',
            num_classes=len(self.classes),
            k=self.k,
            emb_dims=self.emb_dims,
            dropout=self.dropout
        ).to(self.device)

        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ Modelo cargado desde: {checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"   Accuracy de validación: {checkpoint['val_acc']:.2f}%")

    def predict_single(self, points):
        """
        Predice una única muestra

        Args:
            points: array numpy [N, 3]

        Returns:
            pred_class: clase predicha (string)
            confidence: confianza [0-1]
            probs: probabilidades para cada clase
        """
        # Convertir a tensor
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(points_tensor)
                probs = F.softmax(outputs, dim=1)

        pred_idx = outputs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        pred_class = self.classes[pred_idx]

        return pred_class, confidence, probs[0].cpu().numpy()

    def predict_batch(self, points_list):
        """
        Predice múltiples muestras en batch

        Args:
            points_list: lista de arrays numpy [N, 3]

        Returns:
            predictions: lista de (clase, confianza, probabilidades)
        """
        # Convertir a tensor batch
        points_tensor = torch.stack([
            torch.from_numpy(p).float() for p in points_list
        ]).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(points_tensor)
                probs = F.softmax(outputs, dim=1)

        pred_indices = outputs.argmax(dim=1).cpu().numpy()
        confidences = probs.max(dim=1)[0].cpu().numpy()
        all_probs = probs.cpu().numpy()

        predictions = []
        for i in range(len(points_list)):
            pred_class = self.classes[pred_indices[i]]
            confidence = confidences[i]
            class_probs = all_probs[i]
            predictions.append((pred_class, confidence, class_probs))

        return predictions

    def predict_files(self, file_paths, from_dump=False, batch_size=32):
        """
        Predice múltiples archivos

        Args:
            file_paths: lista de rutas a archivos
            from_dump: True si son archivos dump
            batch_size: tamaño del batch para inferencia

        Returns:
            results: lista de diccionarios con resultados
        """
        results = []
        points_batch = []
        files_batch = []

        pbar = tqdm(file_paths, desc='Prediciendo')

        for filepath in pbar:
            # Cargar puntos
            try:
                if from_dump:
                    points = load_dump_surface(filepath, self.n_points)
                else:
                    points = load_off(filepath, self.n_points)
            except Exception as e:
                print(f"⚠️  Error en {filepath.name}: {e}")
                continue

            points_batch.append(points)
            files_batch.append(filepath)

            # Predecir cuando el batch está lleno
            if len(points_batch) >= batch_size:
                predictions = self.predict_batch(points_batch)

                for filepath, (pred_class, confidence, probs) in zip(files_batch, predictions):
                    results.append({
                        'file': filepath.name,
                        'prediction': pred_class,
                        'confidence': float(confidence),
                        **{f'prob_{c}': float(p) for c, p in zip(self.classes, probs)}
                    })

                    # Update progress bar
                    pbar.set_postfix({'pred': pred_class, 'conf': f'{confidence:.2f}'})

                points_batch = []
                files_batch = []

        # Predecir último batch
        if points_batch:
            predictions = self.predict_batch(points_batch)

            for filepath, (pred_class, confidence, probs) in zip(files_batch, predictions):
                results.append({
                    'file': filepath.name,
                    'prediction': pred_class,
                    'confidence': float(confidence),
                    **{f'prob_{c}': float(p) for c, p in zip(self.classes, probs)}
                })

        return results


def main():
    parser = argparse.ArgumentParser(description='DGCNN Prediction - GPU Optimized')

    parser.add_argument('--model', type=str, required=True,
                        help='Directorio del modelo entrenado')
    parser.add_argument('--input', type=str, required=True,
                        help='Archivo o directorio de entrada')
    parser.add_argument('--from_dump', action='store_true',
                        help='Entrada son archivos dump de LAMMPS')
    parser.add_argument('--output', type=str, default=None,
                        help='Archivo CSV de salida')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch para inferencia')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Dispositivo (por defecto: auto)')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Deshabilitar mixed precision')

    args = parser.parse_args()

    # Crear predictor
    predictor = Predictor(
        model_dir=args.model,
        device=args.device,
        mixed_precision=not args.no_mixed_precision
    )

    # Determinar archivos a predecir
    input_path = Path(args.input)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        if args.from_dump:
            # Buscar archivos dump
            files = [f for f in input_path.iterdir()
                    if f.is_file() and (f.suffix == '.dump' or f.suffix == '')]
        else:
            # Buscar archivos .off
            files = list(input_path.glob('*.off'))
        files = sorted(files)
    else:
        print(f"❌ Error: {args.input} no existe")
        return

    if len(files) == 0:
        print(f"❌ No se encontraron archivos en {args.input}")
        return

    print(f"\n📁 Archivos a predecir: {len(files)}\n")

    # Predecir
    start_time = time.time()

    results = predictor.predict_files(
        files,
        from_dump=args.from_dump,
        batch_size=args.batch_size
    )

    elapsed = time.time() - start_time

    # Mostrar resultados
    print("\n" + "=" * 70)
    print("📊 RESULTADOS")
    print("=" * 70)

    for result in results:
        print(f"{result['file']:40} → {result['prediction']:20} ({result['confidence']*100:5.1f}%)")

    # Estadísticas
    print("\n" + "-" * 70)
    print("📈 ESTADÍSTICAS")
    print("-" * 70)
    print(f"Total archivos: {len(results)}")
    print(f"Tiempo total: {elapsed:.2f}s")
    print(f"Tiempo promedio: {elapsed/len(results):.3f}s por archivo")
    print(f"Throughput: {len(results)/elapsed:.1f} archivos/seg")

    # Distribución de predicciones
    print("\nDistribución de predicciones:")
    from collections import Counter
    pred_counts = Counter(r['prediction'] for r in results)
    for class_name in predictor.classes:
        count = pred_counts.get(class_name, 0)
        pct = 100.0 * count / len(results) if results else 0
        print(f"  {class_name}: {count} ({pct:.1f}%)")

    # Guardar resultados
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\n💾 Resultados guardados en: {args.output}")

    print("\n" + "=" * 70)
    print("✅ PREDICCIÓN COMPLETADA")
    print("=" * 70)


if __name__ == '__main__':
    main()
