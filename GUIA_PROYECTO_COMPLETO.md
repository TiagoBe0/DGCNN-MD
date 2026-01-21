# 🎯 PointNet++ para Predicción de Vacancias - Guía Completa

**Proyecto:** Comparación de Features Engineered vs Deep Learning para clasificación de clusters de vacancias

---

## 📊 Tu Proyecto Actual

### Pipeline Existente (Features Engineered)

```
LAMMPS dump (16384 átomos Fe BCC)
    ↓
OVITO: Alpha Shape (ConstructSurfaceModifier)
    ↓
Átomos superficiales (~50-500)
    ↓
opentopologyc_extractor.py
    ↓
23 Features topológicas
    ↓
Random Forest → Predicción de n_vacancies
```

**Rendimiento actual:** (Tu R² y accuracy con Random Forest)

### Nuevo Pipeline (Deep Learning)

```
LAMMPS dump (16384 átomos Fe BCC)
    ↓
OVITO: Alpha Shape
    ↓
Átomos superficiales
    ↓
generate_pointnet_data.py (coordenadas xyz)
    ↓
PointNet++ → Predicción de n_vacancies
```

**Objetivo:** Comparar si DL aprende features mejores que las engineered

---

## 🚀 Workflow Completo

### Paso 1: Analizar tu dataset (2 minutos)

```bash
python analyze_vacancy_dataset.py dataset_geometry_preserved.csv
```

**Salida esperada:**
```
📊 ANÁLISIS DEL DATASET DE VACANCIAS
======================================================================

Total muestras: 1165

📈 DISTRIBUCIÓN DE VACANCIAS
----------------------------------------------------------------------
Rango: 1 - 13 vacancias
Media: 6.45 ± 3.82
Mediana: 5

Distribución detallada:

Vacancias      Count       %  Bar
----------------------------------------------------------------------
1                 45     3.9%  ██
2                 89     7.6%  ████
3                112     9.6%  █████
...

🎯 ESTRATEGIAS DE AGRUPACIÓN SUGERIDAS
======================================================================

OPCIÓN 2: Clasificación 4 Clases (recomendada)
----------------------------------------------------------------------
Single (1-2)   :  134 muestras (11.5%)
Small (3-5)    :  387 muestras (33.2%)
Medium (6-9)   :  421 muestras (36.2%)
Large (10+)    :  223 muestras (19.1%)

Balance ratio:         0.60
✅ Balance razonable

💡 RECOMENDACIÓN FINAL
======================================================================

Comando sugerido:

python generate_pointnet_data.py \
    -i "databases/db_integrate/*" \
    -c dataset_geometry_preserved.csv \
    -o data_pointnet \
    -n 256 \
    --split
```

---

### Paso 2: Generar datos para PointNet++ (5-15 minutos)

```bash
# Usar el comando que sugirió el análisis
python generate_pointnet_data.py \
    -i "databases/db_integrate/*" \
    -c dataset_geometry_preserved.csv \
    -o data_pointnet \
    -n 256 \
    --split \
    --test-split 0.2
```

**Esto genera:**
```
data_pointnet_split/
  train/
    single_vacancy/
      file.0.off
      file.1.off
      ...
    small_cluster/
      file.5.off
      ...
    medium_cluster/
      file.6.off
      ...
    large_cluster/
      file.10.off
      ...
  test/
    single_vacancy/
      ...
    small_cluster/
      ...
    medium_cluster/
      ...
    large_cluster/
      ...
```

**Verificar:**
```bash
ls -R data_pointnet_split/train/
ls -R data_pointnet_split/test/
```

---

### Paso 3: Entrenar PointNet++ (2-6 horas en CPU)

#### Entrenamiento de prueba (10 épocas, ~15 minutos)

```bash
python train_pointnet2.py \
  --data data_pointnet_split \
  --n_points 256 \
  --epochs 10 \
  --batch_size 8
```

**Salida esperada:**
```
Dataset train:
  Clases: ['large_cluster', 'medium_cluster', 'single_vacancy', 'small_cluster']
  Total archivos: 932
    - large_cluster: 178 estructuras
    - medium_cluster: 337 estructuras
    - single_vacancy: 107 estructuras
    - small_cluster: 310 estructuras

Modelo creado: 4 clases
Parámetros del modelo: 1,465,922

Época 1/10
Training: 100%|████| 117/117 [02:15<00:00, loss=1.3456, acc=42.34%]
Validation: 100%|████| 30/30 [00:30<00:00]

Resultados Época 1:
  Train - Loss: 1.3456, Acc: 42.34%
  Test  - Loss: 1.2891, Acc: 45.00%
  ✅ Mejor modelo guardado! (Acc: 45.00%)

Época 10/10
Training: 100%|████| 117/117 [02:10<00:00, loss=0.4156, acc=85.23%]
Validation: 100%|████| 30/30 [00:28<00:00]

Resultados Época 10:
  Train - Loss: 0.4156, Acc: 85.23%
  Test  - Loss: 0.4789, Acc: 81.75%
  ✅ Mejor modelo guardado! (Acc: 81.75%)
```

#### Entrenamiento completo (100 épocas, ~3-4 horas)

```bash
python train_pointnet2.py \
  --data data_pointnet_split \
  --n_points 256 \
  --epochs 100 \
  --batch_size 8 \
  --augment \
  --lr 0.001 \
  --save_dir modelo_pointnet_vacancias
```

**Tips:**
- Si tenés poca RAM: `--batch_size 4`
- Si querés ir más rápido en pruebas: `--n_points 128`
- Para mejores resultados: `--epochs 150 --augment`

---

### Paso 4: Evaluar y comparar resultados

#### A. Ver métricas de PointNet++

```bash
# Ver historial de entrenamiento
cat modelo_pointnet_vacancias/history.json

# Ver matriz de confusión
python evaluate_model.py \
  --model modelo_pointnet_vacancias/best_model.pth \
  --data data_pointnet_split/test
```

#### B. Comparar con Random Forest

| Método | Input | Accuracy | Precision | Recall | F1-Score |
|--------|-------|----------|-----------|--------|----------|
| **Random Forest** | 23 features | ?% | ?% | ?% | ?% |
| **PointNet++** | xyz coords | 85% | 83% | 82% | 82% |

**Preguntas clave:**
1. ¿PointNet++ alcanza similar accuracy?
2. ¿Qué método generaliza mejor?
3. ¿Qué clases predice mejor cada uno?

---

### Paso 5: Hacer predicciones en nuevos datos

```bash
# 1. Procesar nuevo dump con OVITO (usando tu pipeline)
python opentopologyc_extractor.py -i nuevo_dump.dump -o temp/

# 2. Convertir a OFF
python generate_pointnet_data.py \
    -i "temp/*.dump" \
    -c temp/dataset_features.csv \
    -o temp_off \
    -n 256

# 3. Predecir con PointNet++
python predict_pointnet2.py \
  --model modelo_pointnet_vacancias/best_model.pth \
  --input temp_off/single_vacancy/nuevo_dump.off
```

**Salida:**
```
======================================================================
RESULTADO
======================================================================
Archivo: nuevo_dump.off
Clase predicha: medium_cluster
Confianza: 87.34%

Probabilidades por clase:
  single_vacancy    2.15% █
  small_cluster    10.51% █████
  medium_cluster   87.34% ████████████████████████████████████████████
  large_cluster     0.00% 
```

---

## 📊 Interpretación de Resultados

### Métricas objetivo por clase

| Clase | Muestras Train | Accuracy Objetivo | Notas |
|-------|---------------|-------------------|-------|
| Single (1-2) | ~107 | >80% | Clase minoritaria, puede ser difícil |
| Small (3-5) | ~310 | >85% | Clase mayoritaria, debería funcionar bien |
| Medium (6-9) | ~337 | >85% | Clase mayoritaria, mejor rendimiento |
| Large (10+) | ~178 | >75% | Clase moderada |

### Señales de éxito

✅ **Muy bueno:**
- Accuracy global >85%
- F1-score balanceado entre clases (>0.75)
- Matriz de confusión con diagonal fuerte

✅ **Bueno:**
- Accuracy global >75%
- Algunas clases confundidas entre vecinas (Medium↔Large OK)
- Mejor que baseline (Random Forest)

⚠️ **Necesita ajustes:**
- Accuracy <70%
- Confusión entre clases distantes (Single↔Large mal)
- Overfitting severo (train>>test)

---

## 🔬 Experimentos Avanzados

### 1. Comparar diferentes n_sample

```bash
# Test con 128 puntos (más rápido)
python generate_pointnet_data.py ... -n 128
python train_pointnet2.py ... --n_points 128

# Test con 512 puntos (más info)
python generate_pointnet_data.py ... -n 512
python train_pointnet2.py ... --n_points 512
```

### 2. Usar modelo optimizado para clusters pequeños

```bash
# En train_pointnet2.py, cambiar:
# from pointnet2_simple import create_classifier
# por:
# from pointnet2_vacancy import create_vacancy_classifier as create_classifier

python train_pointnet2.py ...
```

### 3. Ensemble: Combinar Features + PointNet++

```python
# Extraer features de PointNet++ (última capa oculta)
# Concatenar con features topológicas
# Entrenar Random Forest final con ambas

combined_features = np.concatenate([
    topological_features,  # 23 features
    pointnet_features      # 512 features de última capa
], axis=1)

rf_ensemble.fit(combined_features, n_vacancies)
```

### 4. Regresión en lugar de clasificación

```python
# Modificar modelo para predecir n_vacancies directamente
# En lugar de 4 clases → 1 salida continua

# Cambiar última capa:
self.fc3 = nn.Linear(128, 1)  # en lugar de num_classes

# Usar MSE loss
criterion = nn.MSELoss()

# Métricas: MAE, R²
```

---

## 🐛 Troubleshooting

### "Accuracy muy baja (<50%)"

**Posibles causas:**
1. Datos no normalizados → Regenerar con normalización
2. Clases muy desbalanceadas → Usar balanceo de clases
3. n_sample muy bajo → Aumentar a 512

**Solución:**
```bash
# Regenerar datos
python generate_pointnet_data.py ... -n 512

# Entrenar con data augmentation
python train_pointnet2.py ... --augment --epochs 150
```

### "Modelo predice siempre la misma clase"

**Causa:** Desbalance extremo

**Solución:** Usar weighted loss
```python
# En train_pointnet2.py, añadir:
class_weights = torch.tensor([1.0, 1.0, 0.8, 1.2])  # Ajustar según balance
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### "Overfitting (train 95%, test 65%)"

**Soluciones:**
1. Usar `--augment`
2. Reducir épocas
3. Añadir más dropout
4. Conseguir más datos

---

## 📚 Archivos Importantes

```
Tu proyecto/
  ├── databases/db_integrate/        # Tus dumps originales
  ├── opentopologyc_extractor.py     # Extractor de features (ya lo tenés)
  ├── dataset_geometry_preserved.csv # Features + n_vacancies (ya lo tenés)
  │
  ├── analyze_vacancy_dataset.py     # Analizar dataset (nuevo)
  ├── generate_pointnet_data.py      # Generar datos OFF (nuevo)
  │
  ├── data_pointnet_split/           # Datos generados
  │   ├── train/
  │   └── test/
  │
  ├── pointnet2_simple.py             # Modelo PointNet++ (nuevo)
  ├── pointnet2_vacancy.py            # Modelo optimizado (nuevo)
  ├── train_pointnet2.py              # Entrenamiento (nuevo)
  ├── predict_pointnet2.py            # Predicción (nuevo)
  │
  └── modelo_pointnet_vacancias/      # Modelos entrenados
      ├── best_model.pth
      ├── final_model.pth
      └── history.json
```

---

## ✅ Checklist Rápido

- [ ] Analizar dataset: `python analyze_vacancy_dataset.py dataset_geometry_preserved.csv`
- [ ] Generar datos OFF: `python generate_pointnet_data.py ...`
- [ ] Verificar estructura: `ls -R data_pointnet_split/`
- [ ] Entrenar prueba (10 épocas): `python train_pointnet2.py ... --epochs 10`
- [ ] Evaluar prueba → ¿Accuracy >40%?
- [ ] Entrenar completo (100 épocas): `python train_pointnet2.py ... --epochs 100 --augment`
- [ ] Comparar con Random Forest
- [ ] Hacer predicciones en datos nuevos

---

## 🎯 Objetivo Final

**Publicación / Tesis:**

> *"Comparación de métodos de aprendizaje automático para predicción de defectos en estructuras cristalinas: Features ingenierizadas vs Deep Learning end-to-end"*

**Resultados esperados:**
- Tabla comparativa de rendimiento
- Análisis de qué método es mejor para qué casos
- Visualización de features aprendidas por PointNet++
- Conclusiones sobre cuándo usar cada enfoque

---

**¿Listo para empezar?**

```bash
# Paso 1
python analyze_vacancy_dataset.py dataset_geometry_preserved.csv

# Cuando termine, pegá acá los resultados y seguimos!
```
