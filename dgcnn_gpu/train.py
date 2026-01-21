#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGCNN Training Script - GPU Optimized
======================================

Script de entrenamiento optimizado para GPU con:
- Mixed Precision Training (AMP)
- Multi-GPU support (DataParallel/DistributedDataParallel)
- Gradient Accumulation
- Learning Rate Scheduling
- Early Stopping
- TensorBoard logging

Uso:
    # Entrenamiento básico
    python train.py --data_dir data/ --classes vacancy cluster perfect defect

    # Con mixed precision y multi-GPU
    python train.py --data_dir data/ --classes vacancy cluster --mixed_precision --multi_gpu

    # Con gradient accumulation para batch grande
    python train.py --data_dir data/ --batch_size 16 --accumulation_steps 4  # batch efectivo = 64
"""

import argparse
import sys
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Añadir directorios al path
sys.path.append(str(Path(__file__).parent.parent))

from dgcnn_gpu.model import get_model
from utils.data_utils import PointCloudDataset, DumpDataset, get_dataloader, save_config


class Trainer:
    """Clase para entrenar DGCNN en GPU"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Crear directorio de salida
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

        # Datasets
        print("=" * 70)
        print("🔷 DGCNN Training - GPU Optimized")
        print("=" * 70)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed Precision: {args.mixed_precision}")
        print(f"Multi-GPU: {args.multi_gpu and torch.cuda.device_count() > 1}")
        print("-" * 70)

        self._load_data()

        # Modelo
        self.model = get_model(
            task='classification',
            num_classes=len(args.classes),
            k=args.k,
            emb_dims=args.emb_dims,
            dropout=args.dropout
        ).to(self.device)

        # Multi-GPU
        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler
        if args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

        # Tracking
        self.best_acc = 0.0
        self.patience_counter = 0
        self.global_step = 0

        # Guardar config
        self._save_config()

    def _load_data(self):
        """Carga datasets"""
        args = self.args

        print("\n📁 Cargando datos...")

        if args.from_dump:
            DatasetClass = DumpDataset
        else:
            DatasetClass = PointCloudDataset

        # Training set
        train_dataset = DatasetClass(
            data_dir=Path(args.data_dir) / 'train',
            classes=args.classes,
            n_points=args.n_points,
            use_augmentation=True,
            cache_data=args.cache_data
        )

        # Validation set
        val_dataset = DatasetClass(
            data_dir=Path(args.data_dir) / 'val',
            classes=args.classes,
            n_points=args.n_points,
            use_augmentation=False,
            cache_data=args.cache_data
        )

        # DataLoaders
        self.train_loader = get_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = get_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        print(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    def _save_config(self):
        """Guarda configuración"""
        config = {
            'classes': self.args.classes,
            'n_points': self.args.n_points,
            'k': self.args.k,
            'emb_dims': self.args.emb_dims,
            'dropout': self.args.dropout,
            'batch_size': self.args.batch_size,
            'lr': self.args.lr,
            'epochs': self.args.epochs,
            'mixed_precision': self.args.mixed_precision,
            'multi_gpu': self.args.multi_gpu
        }

        save_config(self.output_dir / 'config.json', config)

    def train_epoch(self, epoch):
        """Entrena una época"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')

        for batch_idx, (points, labels) in enumerate(pbar):
            points, labels = points.to(self.device), labels.to(self.device)

            # Forward pass con mixed precision
            with torch.cuda.amp.autocast(enabled=self.args.mixed_precision):
                outputs = self.model(points)
                loss = self.criterion(outputs, labels)

                # Gradient accumulation
                loss = loss / self.args.accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (cada accumulation_steps)
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Métricas
            total_loss += loss.item() * self.args.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{acc:.2f}%'
            })

            # TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/acc', acc, self.global_step)

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        """Valida el modelo"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Confusion matrix
        num_classes = len(self.args.classes)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        for points, labels in tqdm(self.val_loader, desc='Validation'):
            points, labels = points.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.mixed_precision):
                outputs = self.model(points)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update confusion matrix
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[t, p] += 1

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total

        # Per-class accuracy
        per_class_acc = confusion.diagonal() / confusion.sum(axis=1)

        return val_loss, val_acc, per_class_acc, confusion

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Guarda checkpoint"""
        model_state_dict = self.model.module.state_dict() if isinstance(
            self.model, nn.DataParallel
        ) else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': self.args.classes
        }

        # Último checkpoint
        torch.save(checkpoint, self.output_dir / 'last_model.pth')

        # Mejor checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"💾 Mejor modelo guardado (acc: {val_acc:.2f}%)")

    def train(self):
        """Loop de entrenamiento principal"""
        print("\n🚀 Iniciando entrenamiento...")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, per_class_acc, confusion = self.validate()

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Per-class accuracy
            print("  Per-class accuracy:")
            for i, class_name in enumerate(self.args.classes):
                print(f"    {class_name}: {per_class_acc[i]*100:.2f}%")

            # TensorBoard
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/acc', val_acc, epoch)
            self.writer.add_scalar('lr', current_lr, epoch)

            # Save checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_acc, is_best)

            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\n⚠️  Early stopping (paciencia: {self.args.patience})")
                break

        # Resumen final
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"⏱️  Tiempo total: {elapsed/60:.1f} min")
        print(f"🏆 Mejor accuracy: {self.best_acc:.2f}%")
        print(f"📂 Modelos guardados en: {self.output_dir}")
        print("=" * 70)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='DGCNN Training - GPU Optimized')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directorio de datos (debe tener train/ y val/)')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Nombres de las clases')
    parser.add_argument('--n_points', type=int, default=256,
                        help='Número de puntos por muestra')
    parser.add_argument('--from_dump', action='store_true',
                        help='Cargar desde archivos dump (requiere OVITO)')
    parser.add_argument('--cache_data', action='store_true',
                        help='Cachear datos en memoria')

    # Model
    parser.add_argument('--k', type=int, default=20,
                        help='Número de vecinos en k-NN')
    parser.add_argument('--emb_dims', type=int, default=1024,
                        help='Dimensión del embedding')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='LR scheduler')

    # GPU Optimization
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Usar mixed precision (AMP)')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Usar múltiples GPUs (DataParallel)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Número de workers para DataLoader')

    # Other
    parser.add_argument('--output_dir', type=str, default='outputs/dgcnn_gpu',
                        help='Directorio de salida')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    args = parser.parse_args()

    # Train
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
