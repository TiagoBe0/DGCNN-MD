#!/bin/bash
# Script de instalación de PointNet++ para CPU
# Autor: Santiago - Análisis de simulaciones MD

set -e  # Sale si hay error

echo "=== Instalación PointNet++ para Análisis de Estructuras Atómicas ==="
echo ""

# 1. Crear entorno virtual
echo "[1/6] Creando entorno virtual Python..."
python3 -m venv pointnet2_env
source pointnet2_env/bin/activate

# 2. Actualizar pip
echo "[2/6] Actualizando pip..."
pip install --upgrade pip

# 3. Instalar PyTorch (versión CPU)
echo "[3/6] Instalando PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar dependencias científicas
echo "[4/6] Instalando dependencias científicas..."
pip install numpy scipy matplotlib h5py tqdm

# 5. Clonar e instalar PointNet++
echo "[5/6] Clonando repositorio PointNet++..."
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
cd Pointnet_Pointnet2_pytorch

# 6. Verificar instalación
echo "[6/6] Verificando instalación..."
python -c "import torch; print(f'PyTorch {torch.__version__} instalado correctamente')"
python -c "import numpy; print(f'NumPy {numpy.__version__} instalado correctamente')"

echo ""
echo "✅ ¡Instalación completada!"
echo ""
echo "Para activar el entorno en el futuro:"
echo "  source pointnet2_env/bin/activate"
echo ""
echo "Para usar PointNet++:"
echo "  cd Pointnet_Pointnet2_pytorch"
echo ""
