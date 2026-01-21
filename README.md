# PointNet++ para Análisis de Estructuras Atómicas

**Autor:** Santiago  
**Objetivo:** Adaptar PointNet++ para analizar simulaciones de dinámica molecular (LAMMPS)

---

## 📋 ¿Qué es esto?

Este proyecto adapta **PointNet++** (red neuronal profunda para nubes de puntos 3D) para analizar estructuras atómicas de simulaciones LAMMPS. 

### Casos de uso
- ✅ Clasificar estructuras con/sin defectos
- ✅ Detectar vacancias en cristales
- ✅ Identificar tipos de estructura cristalina
- ✅ Segmentar regiones defectuosas en la estructura
- ✅ Predecir propiedades a partir de geometría atómica

---

## 🚀 Instalación Rápida

### Requisitos previos
- Python 3.7 o superior
- Linux (Ubuntu/Debian recomendado)
- ~2 GB de espacio libre

### Instalación automática

```bash
# 1. Hacer ejecutable el instalador
chmod +x install_pointnet2.sh

# 2. Instalar todo
./install_pointnet2.sh

# 3. Activar entorno
source pointnet2_env/bin/activate
```

### Instalación manual (si prefieres paso a paso)

```bash
# 1. Crear entorno virtual
python3 -m venv pointnet2_env
source pointnet2_env/bin/activate

# 2. Instalar PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Instalar dependencias
pip install numpy scipy matplotlib h5py tqdm

# 4. Clonar PointNet++
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git

# 5. Verificar
python -c "import torch; print('PyTorch:', torch.__version__)"
```

---

## 📂 Estructura del Proyecto

```
.
├── install_pointnet2.sh          # Script de instalación
├── dump_to_off.py                # Convertidor LAMMPS → OFF
├── train_atomic_example.py       # Ejemplos de entrenamiento
├── pointnet2_env/                # Entorno virtual Python
└── Pointnet_Pointnet2_pytorch/   # Código de PointNet++
    ├── models/                   # Modelos (clasificación, segmentación)
    ├── data/                     # Aquí van tus datos
    └── train_classification.py   # Script de entrenamiento
```

---

## 🔄 Flujo de Trabajo

### 1️⃣ Convertir archivos LAMMPS a formato OFF

El formato `.off` es un formato geométrico simple que PointNet++ entiende.

```bash
# Convertir último timestep
python dump_to_off.py estructura.dump estructura.off

# Convertir timestep específico
python dump_to_off.py estructura.dump estructura.off --timestep 0

# Normalizar coordenadas (recomendado para ML)
python dump_to_off.py estructura.dump estructura.off --normalize

# Muestrear 1024 puntos (PointNet++ usa 1024 o 2048)
python dump_to_off.py estructura.dump estructura.off --n-sample 1024

# Filtrar solo átomos de Fe (tipo 1)
python dump_to_off.py estructura.dump estructura.off --atom-types 1

# Ejemplo completo
python dump_to_off.py mi_simulacion.dump salida.off \
  --normalize \
  --n-sample 1024 \
  --atom-types 1 2
```

### 2️⃣ Organizar datos para entrenamiento

PointNet++ necesita datos organizados por clases:

```
data/
  train/
    sin_vacancias/
      estructura_001.off
      estructura_002.off
      ...
    con_vacancias/
      estructura_001.off
      estructura_002.off
      ...
  test/
    sin_vacancias/
      estructura_050.off
      ...
    con_vacancias/
      estructura_050.off
      ...
```

**Script de ayuda para conversión batch:**

```bash
#!/bin/bash
# convertir_batch.sh

# Crear directorios
mkdir -p data/train/sin_vacancias
mkdir -p data/train/con_vacancias

# Convertir dumps sin vacancias
for dump in dumps_sin_vacancias/*.dump; do
  nombre=$(basename "$dump" .dump)
  python dump_to_off.py "$dump" "data/train/sin_vacancias/${nombre}.off" \
    --normalize --n-sample 1024
done

# Convertir dumps con vacancias
for dump in dumps_con_vacancias/*.dump; do
  nombre=$(basename "$dump" .dump)
  python dump_to_off.py "$dump" "data/train/con_vacancias/${nombre}.off" \
    --normalize --n-sample 1024
done

echo "✅ Conversión completada!"
```

### 3️⃣ Entrenar el modelo

```bash
cd Pointnet_Pointnet2_pytorch

# Entrenar clasificador (por defecto usa ModelNet40)
python train_classification.py \
  --model pointnet2_cls_ssg \
  --log_dir pointnet2_atomic \
  --epoch 100 \
  --batch_size 8

# Con tus propios datos (necesitas adaptar el código)
# Ver train_atomic_example.py para ejemplos
```

### 4️⃣ Hacer predicciones

```bash
# Convertir nueva estructura
python dump_to_off.py nueva_estructura.dump nueva.off --normalize --n-sample 1024

# Predecir (necesitas implementar el script de predicción)
# Ver train_atomic_example.py para ejemplos
```

---

## 🎯 Ejemplos de Uso

### Ejemplo 1: Clasificar estructuras BCC vs FCC

```bash
# 1. Preparar datos
mkdir -p data/train/{bcc,fcc}

# 2. Convertir dumps BCC
for f in simulaciones_bcc/*.dump; do
  python dump_to_off.py "$f" "data/train/bcc/$(basename $f .dump).off" \
    --normalize --n-sample 2048
done

# 3. Convertir dumps FCC
for f in simulaciones_fcc/*.dump; do
  python dump_to_off.py "$f" "data/train/fcc/$(basename $f .dump).off" \
    --normalize --n-sample 2048
done

# 4. Entrenar
cd Pointnet_Pointnet2_pytorch
python train_classification.py --model pointnet2_cls_ssg --num_category 2
```

### Ejemplo 2: Detectar concentración de vacancias

```bash
# Organizar por concentración
mkdir -p data/train/{0pct,5pct,10pct,20pct}

# Convertir dumps por categoría
for dump in dumps_0pct/*.dump; do
  python dump_to_off.py "$dump" "data/train/0pct/$(basename $dump .dump).off" \
    --normalize --n-sample 1024
done
# (repetir para otras concentraciones)

# Entrenar clasificador multiclase
cd Pointnet_Pointnet2_pytorch
python train_classification.py --model pointnet2_cls_ssg --num_category 4
```

---

## 🔧 Opciones del convertidor dump_to_off.py

```
uso: dump_to_off.py [-h] [--timestep TIMESTEP] 
                     [--atom-types ATOM_TYPES [ATOM_TYPES ...]]
                     [--normalize] [--n-sample N_SAMPLE]
                     input output

Argumentos posicionales:
  input                 Archivo .dump de entrada
  output               Archivo .off de salida

Opciones:
  --timestep TIMESTEP   Timestep a extraer (-1 = último)
  --atom-types TYPES    Tipos de átomo a incluir (ej: 1 2)
  --normalize           Normalizar coordenadas (centrar y escalar)
  --n-sample N          Número de puntos a muestrear
```

---

## 📊 Formatos de Archivo

### Formato LAMMPS .dump (entrada)

```
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1000
ITEM: BOX BOUNDS pp pp pp
-10.0 10.0
-10.0 10.0
-10.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.5 0.0 0.0
...
```

### Formato OFF (salida)

```
OFF
1000 0 0
0.000000 0.000000 0.000000
1.500000 0.000000 0.000000
...
```

---

## 💡 Tips y Mejores Prácticas

### Para obtener mejores resultados

1. **Normalización:** Siempre usa `--normalize` para ML
2. **Muestreo:** PointNet++ funciona bien con 1024 o 2048 puntos
3. **Balance de clases:** Usa igual número de ejemplos por clase
4. **Datos limpios:** Verifica que tus dumps estén correctos antes de convertir

### Solución de problemas comunes

**Error: "Columna no encontrada"**
- Tu dump tiene un formato diferente
- Edita `dump_to_off.py` línea ~100 para ajustar los nombres de columnas

**Error: "No se encontraron timesteps"**
- El archivo dump está vacío o corrupto
- Verifica con: `head -n 50 tu_archivo.dump`

**PointNet++ muy lento en CPU**
- Es normal, las redes neuronales son lentas en CPU
- Reduce `--epoch` a 10-20 para pruebas rápidas
- Considera usar GPU si tienes disponible

---

## 📚 Recursos Adicionales

### Papers originales
- [PointNet](https://arxiv.org/abs/1612.00593) - CVPR 2017
- [PointNet++](https://arxiv.org/abs/1706.02413) - NIPS 2017

### Implementación usada
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

### Para aprender más
- Tutorial de PointNet: https://medium.com/@itberrios6/point-net-from-scratch-78935690e496
- Documentación PyTorch: https://pytorch.org/docs/stable/

---

## 🤝 Integración con tu pipeline existente

Este proyecto puede integrarse con tu software actual de análisis MD:

```python
# En tu pipeline de Python
from dump_to_off import DumpToOFF
import torch
from models.pointnet2_cls_ssg import get_model

# 1. Convertir dump
converter = DumpToOFF('simulation.dump')
converter.read_dump(timestep=-1)
converter.normalize_coordinates()
converter.write_off('temp.off', n_sample=1024)

# 2. Cargar modelo
model = get_model(num_classes=2, normal_channel=False)
model.load_state_dict(torch.load('mi_modelo.pth'))
model.eval()

# 3. Predecir
# ... (implementar carga y predicción)
```

---

## 📝 TODO / Mejoras futuras

- [ ] Script de predicción listo para usar
- [ ] Soporte para features adicionales (velocidades, energías)
- [ ] Integración directa con OVITO
- [ ] Visualización de resultados
- [ ] Pre-procesamiento automático de datasets grandes
- [ ] Métricas de evaluación específicas para estructuras atómicas

---

## 🐛 Reportar problemas

Si encuentras algún error o tienes sugerencias, abre un issue o contacta directamente.

---

## 📄 Licencia

- PointNet++: MIT License (código original)
- Scripts de conversión: Uso libre para investigación

---

**Última actualización:** Enero 2026  
**Autor:** Santiago  
**Propósito:** Ciencia de materiales computacional
# DGCNN-MD
