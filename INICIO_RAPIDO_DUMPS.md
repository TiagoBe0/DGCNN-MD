# 🚀 Inicio Rápido - Entrenamiento con Dumps

Guía rápida para procesar `db_octubre_fcc_ni` y entrenar DGCNN.

## 📝 Pasos Rápidos

### 1️⃣ Copiar tu base de datos

```bash
# Opción A: Si está en otra ubicación
cp -r /ruta/a/db_octubre_fcc_ni ./

# Opción B: Si está en tu máquina local, súbela al servidor
# scp -r /local/db_octubre_fcc_ni usuario@servidor:~/DGCNN-MD/
```

### 2️⃣ Verificar instalación

```bash
python verify_setup.py
```

Si falta algo, instalar:
```bash
pip install ovito torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3️⃣ Procesar base de datos

**Opción automática (recomendada):**
```bash
./quick_process_dumps.sh -i db_octubre_fcc_ni -c vacancy
```

**Opción manual:**
```bash
python process_dump_database.py \
  --input_dir db_octubre_fcc_ni/ \
  --output_dir data/ \
  --classes vacancy \
  --copy
```

### 4️⃣ Verificar datos procesados

```bash
python verify_setup.py --data_dir data/ --classes vacancy
```

### 5️⃣ Entrenar

**Entrenamiento básico:**
```bash
python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy \
  --epochs 50
```

**Entrenamiento optimizado (GPU):**
```bash
python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy \
  --mixed_precision \
  --batch_size 32 \
  --epochs 100
```

### 6️⃣ Monitorear progreso

```bash
# En otra terminal
tensorboard --logdir outputs/dgcnn_dump/logs
```

Abrir navegador en: http://localhost:6006

---

## 🎯 Ejemplos por Caso de Uso

### Caso 1: Solo vacancias (clasificación binaria)

```bash
# Si tienes: vacancy vs no_vacancy
./quick_process_dumps.sh -i db_octubre_fcc_ni -c vacancy,no_vacancy

python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy no_vacancy \
  --mixed_precision \
  --epochs 100
```

### Caso 2: Múltiples tipos de defectos

```bash
# Si tienes: vacancy, cluster, perfect
./quick_process_dumps.sh -i db_octubre_fcc_ni -c vacancy,cluster,perfect

python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy cluster perfect \
  --mixed_precision \
  --batch_size 24 \
  --epochs 150
```

### Caso 3: Alta precisión (más puntos de superficie)

```bash
python train_dump_example.py \
  --data_dir data/ \
  --classes vacancy \
  --n_points 512 \
  --radius 2.5 \
  --smoothing 15 \
  --mixed_precision \
  --batch_size 16 \
  --epochs 150
```

---

## ⚙️ Parámetros Importantes

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `--n_points` | Puntos de superficie | 128, 256, 512, 1024 |
| `--radius` | Radio superficie | 1.5-3.0 (FCC Ni: 2.0-2.5) |
| `--smoothing` | Suavizado | 10-20 (default: 12) |
| `--batch_size` | Tamaño batch | GPU 6GB: 16-24<br>GPU 12GB: 32-48 |
| `--epochs` | Épocas | 50-200 |
| `--lr` | Learning rate | 0.0001-0.001 |

---

## 🔧 Solución de Problemas

### Error: CUDA out of memory
```bash
# Reducir batch size
python train_dump_example.py --batch_size 8 --n_points 128
```

### Error: OVITO no funciona
```bash
pip uninstall ovito
pip install ovito
```

### Archivos no se clasifican automáticamente
```bash
# Renombrar archivos con nombre de clase
for f in mi_carpeta/*.dump; do
  mv "$f" "vacancy_$(basename $f)"
done
```

### Entrenamiento muy lento
```bash
# Activar mixed precision
python train_dump_example.py --mixed_precision

# Cachear datos (solo datasets pequeños < 1000 archivos)
python train_dump_example.py --cache_data
```

---

## 📚 Documentación Completa

- [DUMP_PROCESSING.md](DUMP_PROCESSING.md) - Guía detallada de procesamiento
- [README_GPU.md](README_GPU.md) - Documentación DGCNN GPU completa
- [README.md](README.md) - Documentación general del proyecto

---

## 🆘 Ayuda

```bash
# Verificar setup
python verify_setup.py --help

# Ver opciones de procesamiento
python process_dump_database.py --help

# Ver opciones de entrenamiento
python train_dump_example.py --help
```

---

**Última actualización:** Enero 2025
