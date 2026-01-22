# Procesamiento de Base de Datos Dump para DGCNN

Esta guía te ayudará a procesar tu base de datos `db_octubre_fcc_ni` (o cualquier otra base de datos de archivos dump) para entrenar DGCNN.

## 📋 Contenido

- [Requisitos](#requisitos)
- [Estructura de Datos Esperada](#estructura-de-datos-esperada)
- [Procesamiento de la Base de Datos](#procesamiento-de-la-base-de-datos)
- [Entrenamiento](#entrenamiento)
- [Solución de Problemas](#solución-de-problemas)

---

## ✅ Requisitos

### Software necesario

1. **OVITO** (para extracción de superficies)
   ```bash
   # Instalar OVITO Python module
   pip install ovito
   ```

2. **PyTorch con CUDA** (para GPU)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Dependencias del proyecto**
   ```bash
   pip install -r requirements.txt
   ```

### Hardware recomendado

- GPU NVIDIA con al menos 6GB VRAM
- 16GB+ RAM (especialmente si usas `--cache_data`)
- Espacio en disco para datos procesados

---

## 📁 Estructura de Datos Esperada

El entrenamiento espera la siguiente estructura:

```
data/
  train/
    vacancy/
      estructura_001.dump
      estructura_002.dump
      ...
    cluster/
      estructura_101.dump
      estructura_102.dump
      ...
  val/
    vacancy/
      estructura_050.dump
      ...
    cluster/
      estructura_150.dump
      ...
```

---

## 🔄 Procesamiento de la Base de Datos

### Opción 1: Procesamiento Automático (Recomendado)

Si tus archivos dump tienen el nombre de la clase en el nombre del archivo:

```bash
python process_dump_database.py \
  --input_dir /ruta/a/db_octubre_fcc_ni/ \
  --output_dir data/ \
  --classes vacancy cluster perfect \
  --train_split 0.8 \
  --copy
```

**Parámetros:**
- `--input_dir`: Directorio con tus archivos dump originales
- `--output_dir`: Directorio donde se creará la estructura train/val
- `--classes`: Lista de clases (nombres que aparecen en los archivos)
- `--train_split`: Proporción para entrenamiento (0.8 = 80% train, 20% val)
- `--copy`: Copia archivos en lugar de moverlos (recomendado para preservar originales)

**Ejemplo con nomenclatura típica:**

Si tus archivos se llaman:
- `estructura_vacancy_001.dump`
- `estructura_vacancy_002.dump`
- `estructura_cluster_001.dump`
- `estructura_perfect_001.dump`

El script automáticamente los clasificará:

```bash
python process_dump_database.py \
  --input_dir db_octubre_fcc_ni/ \
  --output_dir data/ \
  --classes vacancy cluster perfect \
  --copy
```

### Opción 2: Estructura ya Organizada

Si tu base de datos ya está organizada por subdirectorios:

```
db_octubre_fcc_ni/
  vacancy/
    001.dump
    002.dump
  cluster/
    001.dump
    002.dump
```

Usa el mismo comando:

```bash
python process_dump_database.py \
  --input_dir db_octubre_fcc_ni/ \
  --output_dir data/ \
  --classes vacancy cluster \
  --copy
```

### Opción 3: Procesamiento Manual

Si necesitas control total, organiza manualmente:

```bash
# 1. Crear estructura
mkdir -p data/train/vacancy data/train/cluster
mkdir -p data/val/vacancy data/val/cluster

# 2. Copiar archivos (80% train, 20% val)
# ... copiar manualmente o con script bash

# 3. Verificar
python -c "from process_dump_database import DumpDatabaseProcessor; \
           p = DumpDatabaseProcessor('.', 'data', ['vacancy', 'cluster']); \
           p.validate_structure()"
```

---

## 🚀 Entrenamiento

### Entrenamiento Básico

```bash
python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy \
  --epochs 50 \
  --batch_size 16
```

### Entrenamiento Optimizado para GPU

```bash
python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy cluster perfect \
  --mixed_precision \
  --batch_size 32 \
  --epochs 100 \
  --n_points 256
```

### Parámetros Importantes

#### Datos
- `--data_dir`: Directorio con train/ y val/
- `--classes`: Lista de clases (debe coincidir con subdirectorios)
- `--n_points`: Número de puntos de superficie a extraer (default: 256)

#### Superficie OVITO
- `--radius`: Radio para construcción de superficie (default: 2.0)
  - Aumentar para superficies más suaves
  - Reducir para capturar más detalle
- `--smoothing`: Nivel de suavizado (default: 12)
  - Mayor = superficie más suave
  - Menor = superficie más detallada

#### Modelo
- `--k`: Vecinos k-NN (default: 20)
- `--emb_dims`: Dimensión embedding (default: 1024)
- `--dropout`: Tasa de dropout (default: 0.5)

#### Training
- `--batch_size`: Tamaño del batch
  - GPU 6GB: 16-24
  - GPU 12GB: 32-48
  - Reducir si hay errores de memoria
- `--epochs`: Número de épocas (default: 100)
- `--lr`: Learning rate (default: 0.001)

#### Optimización GPU
- `--mixed_precision`: Usar AMP (recomendado, 2-3x más rápido)
- `--multi_gpu`: Usar múltiples GPUs si hay disponibles
- `--cache_data`: Cachear datos en RAM (solo para datasets pequeños < 1000 muestras)
- `--num_workers`: Workers para carga de datos (default: 4)

#### Output
- `--output_dir`: Donde guardar modelos (default: outputs/dgcnn_dump)
- `--patience`: Early stopping patience (default: 20)

### Ejemplo Completo

```bash
python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy cluster perfect \
  --n_points 512 \
  --radius 2.5 \
  --smoothing 15 \
  --batch_size 24 \
  --epochs 150 \
  --lr 0.001 \
  --mixed_precision \
  --output_dir outputs/fcc_ni_model
```

---

## 🔍 Solución de Problemas

### Error: "No se encontraron archivos dump clasificados"

**Causa:** Los archivos no tienen el nombre de clase en el nombre

**Solución:** Renombrar archivos o usar estructura de subdirectorios:
```bash
# Renombrar archivos
for f in *.dump; do
  mv "$f" "vacancy_$f"
done

# O usar subdirectorios
mkdir vacancy cluster
mv vacancy*.dump vacancy/
mv cluster*.dump cluster/
```

### Error: "OVITO no está instalado"

**Solución:**
```bash
pip install ovito

# Si falla, instalar desde conda
conda install -c conda-forge ovito
```

### Error: CUDA out of memory

**Solución 1:** Reducir batch size
```bash
python train_dump_example.py --batch_size 8
```

**Solución 2:** Reducir número de puntos
```bash
python train_dump_example.py --n_points 128
```

**Solución 3:** Usar gradient accumulation
```bash
python train_dump_example.py --batch_size 8 --accumulation_steps 4
# Batch efectivo = 8 * 4 = 32
```

### Advertencia: "Error cargando {archivo}"

**Causa:** Archivo dump corrupto o formato incorrecto

**Solución:** Verificar archivo:
```bash
head -n 50 archivo.dump

# Debe tener:
# ITEM: TIMESTEP
# ITEM: NUMBER OF ATOMS
# ITEM: BOX BOUNDS
# ITEM: ATOMS id type x y z
```

### Entrenamiento muy lento

**Solución 1:** Usar mixed precision
```bash
python train_dump_example.py --mixed_precision
```

**Solución 2:** Cachear datos (solo si dataset < 1000 muestras)
```bash
python train_dump_example.py --cache_data
```

**Solución 3:** Aumentar workers
```bash
python train_dump_example.py --num_workers 8
```

### Accuracy no mejora

**Posibles causas:**

1. **Datos desbalanceados**
   ```bash
   # Verificar distribución
   python -c "import json; print(json.load(open('data/dataset_stats.json')))"
   ```

2. **Parámetros de superficie incorrectos**
   - Aumentar/reducir `--radius`
   - Ajustar `--smoothing`

3. **Overfitting**
   - Aumentar dropout: `--dropout 0.6`
   - Aumentar weight decay: `--weight_decay 1e-3`
   - Reducir épocas: `--epochs 50`

4. **Learning rate incorrecto**
   - Reducir: `--lr 0.0001`
   - Usar scheduler adaptativo: `--scheduler plateau`

---

## 📊 Monitoreo del Entrenamiento

### TensorBoard

```bash
# Iniciar TensorBoard
tensorboard --logdir outputs/dgcnn_dump/logs

# Abrir en navegador
http://localhost:6006
```

### Checkpoints

Los modelos se guardan en:
- `outputs/dgcnn_dump/last_model.pth` - Último checkpoint
- `outputs/dgcnn_dump/best_model.pth` - Mejor modelo (mejor accuracy)
- `outputs/dgcnn_dump/config.json` - Configuración

---

## 🎯 Siguiente Paso: Predicción

Una vez entrenado, usar para predicción:

```bash
python dgcnn_gpu/predict.py \
  --model outputs/dgcnn_dump/best_model.pth \
  --input nueva_estructura.dump \
  --from_dump
```

---

## 📖 Referencias

- [README_GPU.md](README_GPU.md) - Documentación completa DGCNN GPU
- [QUICKSTART.md](QUICKSTART.md) - Inicio rápido
- [README.md](README.md) - Documentación general

---

**¿Problemas?** Abre un issue o contacta al desarrollador.
