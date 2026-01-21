#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Utilities para DGCNN-MD
=============================

Utilidades para cargar y procesar datos de simulaciones moleculares
Optimizado para entrenamiento eficiente en GPU
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict


def load_off(filepath, n_points=256):
    """
    Carga archivo OFF y extrae nube de puntos

    Args:
        filepath: ruta al archivo .off
        n_points: número de puntos a samplear

    Returns:
        points: array numpy [n_points, 3] con coordenadas normalizadas
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parsear header
    if lines[0].strip() == 'OFF':
        n_verts = int(lines[1].split()[0])
        start = 2
    else:
        parts = lines[0].split()
        n_verts = int(parts[1]) if len(parts) > 1 else int(lines[1].split()[0])
        start = 1 if len(parts) > 1 else 2

    # Leer vértices
    points = []
    for i in range(start, start + n_verts):
        if i < len(lines):
            coords = [float(x) for x in lines[i].split()[:3]]
            points.append(coords)

    points = np.array(points, dtype=np.float32)

    # Validación
    if len(points) < 3:
        raise ValueError(f"Muy pocos puntos en {filepath}: {len(points)}")

    # Resampleo
    points = resample_points(points, n_points)

    # Normalización
    points = normalize_points(points)

    return points


def load_dump_surface(filepath, n_points=256, radius=2.0, smoothing=12):
    """
    Carga archivo dump de LAMMPS y extrae superficie usando OVITO

    Args:
        filepath: ruta al archivo dump
        n_points: número de puntos a samplear
        radius: radio para construcción de superficie
        smoothing: nivel de suavizado

    Returns:
        points: array numpy [n_points, 3] con coordenadas normalizadas
    """
    try:
        from ovito.io import import_file
        from ovito.modifiers import (ConstructSurfaceModifier,
                                      InvertSelectionModifier,
                                      DeleteSelectedModifier)
    except ImportError:
        raise ImportError(
            "OVITO no está instalado. "
            "Instalar con: pip install ovito"
        )

    # Pipeline de OVITO
    pipeline = import_file(str(filepath))

    # Construir superficie
    pipeline.modifiers.append(ConstructSurfaceModifier(
        radius=radius,
        select_surface_particles=True,
        smoothing_level=smoothing
    ))

    # Eliminar partículas internas
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())

    # Computar
    data = pipeline.compute()
    points = np.array(data.particles['Position'][...], dtype=np.float32)

    # Validación
    if len(points) < 3:
        raise ValueError(f"Muy pocos puntos de superficie en {filepath}: {len(points)}")

    # Resampleo y normalización
    points = resample_points(points, n_points)
    points = normalize_points(points)

    return points


def resample_points(points, n_points):
    """
    Resamplea nube de puntos a tamaño fijo

    Args:
        points: array [N, 3]
        n_points: número objetivo de puntos

    Returns:
        points: array [n_points, 3]
    """
    n = len(points)

    if n >= n_points:
        # Downsample sin reemplazo
        choice = np.random.choice(n, n_points, replace=False)
    else:
        # Upsample con reemplazo
        choice = np.random.choice(n, n_points, replace=True)

    return points[choice, :]


def normalize_points(points):
    """
    Normaliza nube de puntos a esfera unitaria centrada

    Args:
        points: array [N, 3]

    Returns:
        points: array [N, 3] normalizado
    """
    # Centrar en origen
    centroid = points.mean(axis=0)
    points = points - centroid

    # Escalar a esfera unitaria
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    return points


def augment_points(points):
    """
    Data augmentation para nube de puntos

    Args:
        points: array [N, 3]

    Returns:
        points: array [N, 3] aumentado
    """
    # Rotación aleatoria
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    points = points @ rotation_matrix.T

    # Jitter (ruido pequeño)
    noise = np.random.normal(0, 0.02, points.shape).astype(np.float32)
    points = points + noise

    # Scaling aleatorio
    scale = np.random.uniform(0.8, 1.2)
    points = points * scale

    return points


class PointCloudDataset(Dataset):
    """
    Dataset para nubes de puntos desde archivos OFF

    Optimizado para DataLoader con multi-threading
    """
    def __init__(self, data_dir, classes, n_points=256,
                 use_augmentation=False, cache_data=False):
        """
        Args:
            data_dir: directorio raíz con subdirectorios por clase
            classes: lista de nombres de clases
            n_points: número de puntos por muestra
            use_augmentation: aplicar data augmentation
            cache_data: cachear datos en memoria (usar si dataset es pequeño)
        """
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.n_points = n_points
        self.use_augmentation = use_augmentation
        self.cache_data = cache_data

        # Mapeo clase -> índice
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Encontrar todos los archivos
        self.samples = []
        for class_name in classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  Directorio no encontrado: {class_dir}")
                continue

            for filepath in class_dir.glob('*.off'):
                self.samples.append((filepath, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No se encontraron archivos .off en {data_dir}")

        print(f"Dataset cargado: {len(self.samples)} muestras")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for _, c in self.samples if c == idx)
            print(f"  {class_name}: {count} muestras")

        # Cache
        self.cache = {} if cache_data else None
        if cache_data:
            print("Cargando datos en memoria...")
            for i in range(len(self.samples)):
                filepath, label = self.samples[i]
                points = load_off(filepath, self.n_points)
                self.cache[i] = (points, label)
            print("✅ Datos cacheados")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache is not None:
            points, label = self.cache[idx]
            points = points.copy()  # Copiar para no modificar cache
        else:
            filepath, label = self.samples[idx]
            points = load_off(filepath, self.n_points)

        # Augmentation
        if self.use_augmentation:
            points = augment_points(points)

        # Convertir a tensor
        points = torch.from_numpy(points).float()
        label = torch.tensor(label, dtype=torch.long)

        return points, label


class DumpDataset(Dataset):
    """
    Dataset para archivos dump de LAMMPS

    Extrae superficies usando OVITO
    """
    def __init__(self, data_dir, classes, n_points=256,
                 use_augmentation=False, radius=2.0, smoothing=12):
        """
        Args:
            data_dir: directorio raíz con subdirectorios por clase
            classes: lista de nombres de clases
            n_points: número de puntos por muestra
            use_augmentation: aplicar data augmentation
            radius: radio para construcción de superficie
            smoothing: nivel de suavizado
        """
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.n_points = n_points
        self.use_augmentation = use_augmentation
        self.radius = radius
        self.smoothing = smoothing

        # Mapeo clase -> índice
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Encontrar todos los archivos
        self.samples = []
        for class_name in classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  Directorio no encontrado: {class_dir}")
                continue

            # Buscar archivos dump (sin extensión o .dump)
            for filepath in class_dir.iterdir():
                if filepath.is_file() and (filepath.suffix == '.dump' or filepath.suffix == ''):
                    self.samples.append((filepath, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No se encontraron archivos dump en {data_dir}")

        print(f"Dataset cargado: {len(self.samples)} muestras")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for _, c in self.samples if c == idx)
            print(f"  {class_name}: {count} muestras")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        try:
            points = load_dump_surface(
                filepath,
                self.n_points,
                self.radius,
                self.smoothing
            )
        except Exception as e:
            print(f"⚠️  Error cargando {filepath}: {e}")
            # Retornar puntos aleatorios en caso de error
            points = np.random.randn(self.n_points, 3).astype(np.float32)
            points = normalize_points(points)

        # Augmentation
        if self.use_augmentation:
            points = augment_points(points)

        # Convertir a tensor
        points = torch.from_numpy(points).float()
        label = torch.tensor(label, dtype=torch.long)

        return points, label


def get_dataloader(dataset, batch_size=32, shuffle=True,
                   num_workers=4, pin_memory=True, drop_last=False):
    """
    Crea DataLoader optimizado para GPU

    Args:
        dataset: Dataset de PyTorch
        batch_size: tamaño del batch
        shuffle: mezclar datos
        num_workers: número de workers para carga paralela
        pin_memory: usar pinned memory para transferencias GPU más rápidas
        drop_last: descartar último batch incompleto

    Returns:
        dataloader: DataLoader de PyTorch
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0  # Mantener workers vivos
    )


def save_config(filepath, config):
    """Guarda configuración en JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath):
    """Carga configuración desde JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)
