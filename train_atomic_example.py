#!/usr/bin/env python3
"""
Ejemplo de entrenamiento PointNet++ con estructuras atómicas de LAMMPS
Autor: Santiago - Adaptación para análisis MD

Este script demuestra cómo:
1. Cargar datos .off convertidos desde LAMMPS
2. Entrenar PointNet++ para clasificación/segmentación
3. Hacer predicciones en nuevas estructuras
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Agregar path de PointNet++
sys.path.append('./Pointnet_Pointnet2_pytorch')

from models.pointnet2_cls_ssg import get_model as get_classifier
from models.pointnet2_part_seg import get_model as get_segmenter


class AtomicStructureDataset(torch.utils.data.Dataset):
    """
    Dataset para estructuras atómicas en formato OFF
    
    Ejemplo de estructura de carpetas:
    data/
      class_0/
        structure_0.off
        structure_1.off
      class_1/
        structure_0.off
        structure_1.off
    """
    
    def __init__(self, root_dir, n_points=1024, normalize=True):
        """
        Args:
            root_dir: Directorio raíz con subdirectorios por clase
            n_points: Número de puntos a samplear
            normalize: Si normalizar coordenadas
        """
        self.root_dir = Path(root_dir)
        self.n_points = n_points
        self.normalize = normalize
        
        # Cargar archivos
        self.files = []
        self.labels = []
        
        classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_names = [c.name for c in classes]
        
        for class_idx, class_dir in enumerate(classes):
            off_files = list(class_dir.glob('*.off'))
            self.files.extend(off_files)
            self.labels.extend([class_idx] * len(off_files))
        
        print(f"Dataset cargado:")
        print(f"  Clases: {self.class_names}")
        print(f"  Total archivos: {len(self.files)}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Leer archivo OFF
        point_cloud = self.read_off(self.files[idx])
        label = self.labels[idx]
        
        # Samplear puntos
        if len(point_cloud) > self.n_points:
            choice = np.random.choice(len(point_cloud), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(point_cloud), self.n_points, replace=True)
        
        point_cloud = point_cloud[choice, :]
        
        # Normalizar
        if self.normalize:
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            point_cloud = point_cloud / np.max(np.linalg.norm(point_cloud, axis=1))
        
        # Transponer para PyTorch [3, N]
        point_cloud = torch.from_numpy(point_cloud).float().transpose(0, 1)
        label = torch.tensor(label).long()
        
        return point_cloud, label
    
    @staticmethod
    def read_off(filename):
        """Lee archivo OFF y retorna coordenadas"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Primera línea debe ser "OFF"
        assert lines[0].strip() == 'OFF', "Formato OFF inválido"
        
        # Segunda línea: n_vertices n_faces n_edges
        n_vertices = int(lines[1].strip().split()[0])
        
        # Leer vértices
        vertices = []
        for i in range(2, 2 + n_vertices):
            coords = [float(x) for x in lines[i].strip().split()]
            vertices.append(coords[:3])  # Solo x, y, z
        
        return np.array(vertices)


def train_classifier_example():
    """
    Ejemplo de entrenamiento de clasificador
    
    Casos de uso:
    - Clasificar estructuras con/sin defectos
    - Identificar tipo de estructura cristalina
    - Clasificar concentración de vacancias
    """
    print("\n=== Ejemplo: Entrenamiento de Clasificador ===\n")
    
    # Configuración
    n_classes = 2  # Ejemplo: con defecto / sin defecto
    n_points = 1024
    epochs = 50
    batch_size = 8
    learning_rate = 0.001
    
    # Crear modelo
    model = get_classifier(n_classes, normal_channel=False)
    model = model.cpu()  # Usar CPU
    
    print(f"Modelo creado: {n_classes} clases")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Ejemplo de dataset (ajustar ruta)
    # dataset = AtomicStructureDataset('data/my_structures', n_points=n_points)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\n⚠️  Para entrenar, necesitas:")
    print("  1. Convertir tus .dump a .off usando dump_to_off.py")
    print("  2. Organizar archivos en carpetas por clase:")
    print("     data/")
    print("       sin_defectos/")
    print("         estructura_1.off")
    print("         estructura_2.off")
    print("       con_defectos/")
    print("         estructura_1.off")
    print("         estructura_2.off")
    print("  3. Descomentar las líneas del dataset arriba")
    
    # Loop de entrenamiento (comentado hasta tener datos)
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for points, labels in dataloader:
            points = points.cpu()
            labels = labels.cpu()
            
            optimizer.zero_grad()
            pred, _ = model(points)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f} - Acc: {accuracy:.2f}%')
    
    # Guardar modelo
    torch.save(model.state_dict(), 'atomic_structure_classifier.pth')
    print("\n✅ Modelo guardado: atomic_structure_classifier.pth")
    """


def predict_example():
    """
    Ejemplo de predicción en nueva estructura
    """
    print("\n=== Ejemplo: Predicción en Nueva Estructura ===\n")
    
    # Cargar modelo (ajustar n_classes según tu caso)
    n_classes = 2
    model = get_classifier(n_classes, normal_channel=False)
    # model.load_state_dict(torch.load('atomic_structure_classifier.pth'))
    model = model.cpu()
    model.eval()
    
    print("⚠️  Para hacer predicciones:")
    print("  1. Convierte tu .dump a .off:")
    print("     python dump_to_off.py estructura.dump estructura.off --normalize --n-sample 1024")
    print("  2. Carga el modelo entrenado")
    print("  3. Ejecuta la predicción")
    
    # Ejemplo de predicción (comentado)
    """
    # Cargar estructura
    dataset = AtomicStructureDataset('data/test_structures', n_points=1024)
    structure, _ = dataset[0]  # Primera estructura
    structure = structure.unsqueeze(0)  # Batch de 1
    
    # Predecir
    with torch.no_grad():
        pred, _ = model(structure)
        class_idx = torch.argmax(pred, dim=1).item()
        confidence = torch.softmax(pred, dim=1)[0, class_idx].item()
    
    print(f"Predicción: Clase {class_idx}")
    print(f"Confianza: {confidence*100:.2f}%")
    """


def conversion_workflow_example():
    """
    Ejemplo del flujo completo de conversión
    """
    print("\n=== Flujo de Trabajo Completo ===\n")
    
    print("1️⃣  Generar datos de entrenamiento desde LAMMPS:")
    print("    # Convertir múltiples dumps")
    print("    for dump in dumps_sin_defecto/*.dump; do")
    print("      python dump_to_off.py $dump data/sin_defectos/$(basename $dump .dump).off \\")
    print("        --normalize --n-sample 1024")
    print("    done")
    print()
    
    print("2️⃣  Organizar datos:")
    print("    data/")
    print("      train/")
    print("        clase_0/")
    print("        clase_1/")
    print("      test/")
    print("        clase_0/")
    print("        clase_1/")
    print()
    
    print("3️⃣  Entrenar modelo:")
    print("    python train_atomic_classifier.py")
    print()
    
    print("4️⃣  Predecir en nuevas estructuras:")
    print("    python dump_to_off.py nueva_estructura.dump nueva.off --normalize --n-sample 1024")
    print("    python predict_structure.py nueva.off")
    print()


def main():
    print("="*70)
    print("PointNet++ para Análisis de Estructuras Atómicas (LAMMPS)")
    print("="*70)
    
    # Verificar PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: CPU")
    
    # Mostrar ejemplos
    conversion_workflow_example()
    train_classifier_example()
    predict_example()
    
    print("\n" + "="*70)
    print("💡 Próximos pasos:")
    print("="*70)
    print("1. Instalar todo: bash install_pointnet2.sh")
    print("2. Convertir tus dumps: python dump_to_off.py")
    print("3. Entrenar modelo con tus datos")
    print("4. ¡Analizar estructuras atómicas con Deep Learning!")
    print()


if __name__ == '__main__':
    main()
