# 🚀 DGCNN-MD Quick Start

Guía de inicio rápido para entrenar DGCNN en tus datos de simulaciones moleculares.

---

## ⚡ Instalación en 3 Pasos

```bash
# 1. Instalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar
python test_installation.py
```

---

## 📁 Preparar Datos

Organiza tus archivos `.off` en esta estructura:

```
data/
├── train/
│   ├── clase_A/
│   │   ├── sample_001.off
│   │   ├── sample_002.off
│   │   └── ...
│   └── clase_B/
│       └── ...
└── val/
    ├── clase_A/
    └── clase_B/
```

**Ejemplo:** Para clasificar estructuras con/sin vacancias:

```
data/
├── train/
│   ├── no_vacancy/     # 100 archivos .off
│   └── vacancy/        # 100 archivos .off
└── val/
    ├── no_vacancy/     # 20 archivos .off
    └── vacancy/        # 20 archivos .off
```

---

## 🎯 Entrenar Modelo

### Entrenamiento Básico (CPU/GPU automático)

```bash
python dgcnn_gpu/train.py \
    --data_dir data/ \
    --classes no_vacancy vacancy \
    --output_dir outputs/my_first_model
```

### Entrenamiento Optimizado GPU

```bash
python dgcnn_gpu/train.py \
    --data_dir data/ \
    --classes no_vacancy vacancy \
    --batch_size 32 \
    --epochs 200 \
    --mixed_precision \
    --output_dir outputs/gpu_optimized
```

### Monitorear Training

```bash
# En otra terminal
tensorboard --logdir outputs/my_first_model/logs
# Abrir http://localhost:6006
```

---

## 🔮 Hacer Predicciones

### Predecir un Archivo

```bash
python dgcnn_gpu/predict.py \
    --model outputs/my_first_model \
    --input test_sample.off
```

### Predecir Múltiples Archivos

```bash
python dgcnn_gpu/predict.py \
    --model outputs/my_first_model \
    --input test_directory/ \
    --batch_size 64 \
    --output predictions.csv
```

**Salida:**

```
sample_001.off → vacancy      (95.3%)
sample_002.off → no_vacancy   (88.7%)
sample_003.off → vacancy      (92.1%)
```

---

## 📊 Resultados Típicos

| Dataset | Accuracy | Training Time (GPU) |
|---------|----------|---------------------|
| Vacancy Detection (2 classes) | ~95% | 10-20 min |
| Defect Types (4 classes) | ~90% | 20-30 min |
| Crystal Structures (6 classes) | ~85% | 30-45 min |

*Con GPU NVIDIA RTX 3080, 200 epochs*

---

## 💡 Tips Rápidos

### 1. GPU sin Memoria?

```bash
# Reducir batch size
python dgcnn_gpu/train.py --batch_size 8 --mixed_precision ...
```

### 2. Dataset Pequeño?

```bash
# Aumentar epochs y usar cache
python dgcnn_gpu/train.py --epochs 300 --cache_data ...
```

### 3. Clases Desbalanceadas?

Asegúrate de tener similar número de muestras por clase, o implementa class weighting.

---

## 🆘 Problemas Comunes

| Problema | Solución |
|----------|----------|
| `CUDA out of memory` | Reducir `--batch_size` o usar `--mixed_precision` |
| `No .off files found` | Verificar estructura de directorios |
| `Accuracy no mejora` | Aumentar `--epochs`, revisar datos |
| Training muy lento | Verificar GPU disponible, usar `--mixed_precision` |

---

## 📚 Siguiente Paso

Para configuraciones avanzadas, ver:
- **[README_GPU.md](README_GPU.md)**: Documentación completa
- **[README.md](README.md)**: Información general del proyecto

---

## 🎓 Ejemplo Completo

```bash
# 1. Verificar instalación
python test_installation.py

# 2. Organizar datos (ejemplo)
mkdir -p data/{train,val}/{perfect,defective}
# ... copiar archivos .off a directorios correspondientes

# 3. Entrenar
python dgcnn_gpu/train.py \
    --data_dir data/ \
    --classes perfect defective \
    --batch_size 32 \
    --epochs 200 \
    --mixed_precision \
    --output_dir outputs/defect_classifier

# 4. Monitorear (en otra terminal)
tensorboard --logdir outputs/defect_classifier/logs

# 5. Predecir nuevos datos
python dgcnn_gpu/predict.py \
    --model outputs/defect_classifier \
    --input new_samples/ \
    --output results.csv

# 6. Ver resultados
cat results.csv
```

---

**¡Listo para empezar! 🎉**
