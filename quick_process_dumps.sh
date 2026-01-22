#!/bin/bash
# Quick Process Dumps - Script rápido para procesar base de datos dump
# =====================================================================

set -e  # Exit on error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Procesador Rápido de Dumps DGCNN    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuración por defecto
INPUT_DIR="db_octubre_fcc_ni"
OUTPUT_DIR="data"
CLASSES="vacancy"
TRAIN_SPLIT=0.8

# Función de ayuda
show_help() {
    echo "Uso: ./quick_process_dumps.sh [opciones]"
    echo ""
    echo "Opciones:"
    echo "  -i <dir>    Directorio de entrada (default: db_octubre_fcc_ni)"
    echo "  -o <dir>    Directorio de salida (default: data)"
    echo "  -c <clases> Clases separadas por comas (default: vacancy)"
    echo "  -s <split>  Train split 0-1 (default: 0.8)"
    echo "  -h          Mostrar esta ayuda"
    echo ""
    echo "Ejemplo:"
    echo "  ./quick_process_dumps.sh -i mi_db/ -c vacancy,cluster,perfect"
    exit 0
}

# Parsear argumentos
while getopts "i:o:c:s:h" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        c) CLASSES="$OPTARG";;
        s) TRAIN_SPLIT="$OPTARG";;
        h) show_help;;
        *) show_help;;
    esac
done

# Convertir clases de comma-separated a space-separated
CLASSES_SPACE=$(echo $CLASSES | tr ',' ' ')

echo -e "${GREEN}Configuración:${NC}"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Clases: $CLASSES_SPACE"
echo "  Split:  $TRAIN_SPLIT"
echo ""

# Verificar que existe el directorio de entrada
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ Error: Directorio de entrada no existe: $INPUT_DIR${NC}"
    echo ""
    echo "Opciones:"
    echo "  1. Crear el directorio y copiar tus dumps ahí:"
    echo "     mkdir -p $INPUT_DIR"
    echo "     cp /ruta/a/tus/dumps/*.dump $INPUT_DIR/"
    echo ""
    echo "  2. Especificar otro directorio:"
    echo "     ./quick_process_dumps.sh -i /ruta/a/tu/base/datos"
    exit 1
fi

# Contar archivos dump
DUMP_COUNT=$(find "$INPUT_DIR" -name "*.dump" -type f 2>/dev/null | wc -l)
echo -e "${BLUE}📊 Encontrados: $DUMP_COUNT archivos .dump${NC}"

if [ $DUMP_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No se encontraron archivos .dump en $INPUT_DIR${NC}"
    echo ""
    echo "Verifica que:"
    echo "  1. Los archivos tienen extensión .dump"
    echo "  2. Están en el directorio correcto"
    echo ""
    echo "Estructura esperada:"
    echo "  $INPUT_DIR/"
    echo "    estructura_001.dump"
    echo "    estructura_002.dump"
    echo "    ..."
    exit 1
fi

# Preguntar confirmación
echo ""
echo -e "${YELLOW}¿Procesar $DUMP_COUNT archivos? [y/N]${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Cancelado${NC}"
    exit 0
fi

# Procesar
echo ""
echo -e "${GREEN}🔄 Procesando...${NC}"
echo ""

python process_dump_database.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --classes $CLASSES_SPACE \
    --train_split $TRAIN_SPLIT \
    --copy

# Verificar éxito
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ PROCESAMIENTO COMPLETADO${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}📖 Siguiente paso: Entrenar modelo${NC}"
    echo ""
    echo "Entrenamiento básico:"
    echo "  python train_dump_example.py --data_dir $OUTPUT_DIR --classes $CLASSES_SPACE"
    echo ""
    echo "Con GPU optimizado:"
    echo "  python train_dump_example.py --data_dir $OUTPUT_DIR --classes $CLASSES_SPACE --mixed_precision --batch_size 32"
    echo ""
    echo "Ver más opciones:"
    echo "  python train_dump_example.py --help"
    echo ""
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}❌ Error durante el procesamiento${NC}"
    echo ""
    echo "Soluciones:"
    echo "  1. Verificar que process_dump_database.py existe"
    echo "  2. Verificar que Python está instalado"
    echo "  3. Ver el error arriba para más detalles"
    exit 1
fi
