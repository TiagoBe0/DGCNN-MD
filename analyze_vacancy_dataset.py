#!/usr/bin/env python3
"""
Analiza dataset_geometry_preserved.csv para determinar:
- Distribución de n_vacancies
- Tamaño de clusters
- Estrategia de agrupación óptima
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_dataset(csv_path):
    """Analiza dataset y sugiere agrupación"""
    
    print("="*70)
    print("📊 ANÁLISIS DEL DATASET DE VACANCIAS")
    print("="*70)
    print()
    
    # Cargar datos
    df = pd.read_csv(csv_path, index_col='file')
    
    print(f"Total muestras: {len(df)}")
    print()
    
    # Análisis de n_vacancies
    print("📈 DISTRIBUCIÓN DE VACANCIAS")
    print("-"*70)
    
    n_vac = df['n_vacancies']
    
    print(f"Rango: {n_vac.min():.0f} - {n_vac.max():.0f} vacancias")
    print(f"Media: {n_vac.mean():.2f} ± {n_vac.std():.2f}")
    print(f"Mediana: {n_vac.median():.0f}")
    print()
    
    # Histograma
    print("Distribución detallada:")
    print()
    print(f"{'Vacancias':<12} {'Count':>8} {'%':>8}  {'Bar'}")
    print("-"*70)
    
    value_counts = n_vac.value_counts().sort_index()
    
    for vac, count in value_counts.items():
        pct = 100 * count / len(df)
        bar = '█' * int(pct / 2)
        print(f"{vac:<12.0f} {count:8d} {pct:7.1f}%  {bar}")
    
    print()
    
    # Sugerencias de agrupación
    print("="*70)
    print("🎯 ESTRATEGIAS DE AGRUPACIÓN SUGERIDAS")
    print("="*70)
    print()
    
    # Opción 1: Binaria
    print("OPCIÓN 1: Clasificación Binaria (Single vs Cluster)")
    print("-"*70)
    single = (n_vac <= 2).sum()
    cluster = (n_vac > 2).sum()
    print(f"Single vacancy (1-2):  {single:4d} muestras ({100*single/len(df):.1f}%)")
    print(f"Cluster (3+):          {cluster:4d} muestras ({100*cluster/len(df):.1f}%)")
    print(f"Balance ratio:         {min(single, cluster)/max(single, cluster):.2f}")
    if min(single, cluster)/max(single, cluster) > 0.3:
        print("✅ Balance aceptable para clasificación binaria")
    else:
        print("⚠️  Desbalance significativo - considerar estratificación")
    print()
    
    # Opción 2: 4 clases
    print("OPCIÓN 2: Clasificación 4 Clases (recomendada)")
    print("-"*70)
    
    classes = {
        'Single (1-2)': (n_vac >= 1) & (n_vac <= 2),
        'Small (3-5)': (n_vac >= 3) & (n_vac <= 5),
        'Medium (6-9)': (n_vac >= 6) & (n_vac <= 9),
        'Large (10+)': n_vac >= 10
    }
    
    class_counts = {}
    for name, mask in classes.items():
        count = mask.sum()
        class_counts[name] = count
        print(f"{name:15s}: {count:4d} muestras ({100*count/len(df):.1f}%)")
    
    # Calcular balance
    counts = list(class_counts.values())
    balance = min(counts) / max(counts)
    print(f"\nBalance ratio:         {balance:.2f}")
    if balance > 0.2:
        print("✅ Balance razonable")
    else:
        print("⚠️  Algunas clases muy minoritarias")
    print()
    
    # Opción 3: Multi-clase exacta
    print("OPCIÓN 3: Clasificación Multi-clase (Exacta)")
    print("-"*70)
    
    unique_vacs = sorted(value_counts.index)
    n_classes = len(unique_vacs)
    print(f"Número de clases: {n_classes}")
    print()
    
    # Mostrar solo si hay pocas clases
    if n_classes <= 15:
        for vac in unique_vacs:
            count = value_counts[vac]
            print(f"  {vac:2.0f} vacancias: {count:4d} muestras ({100*count/len(df):.1f}%)")
        
        counts_exact = [value_counts[vac] for vac in unique_vacs]
        balance_exact = min(counts_exact) / max(counts_exact)
        print(f"\nBalance ratio:         {balance_exact:.2f}")
        
        if balance_exact > 0.1 and min(counts_exact) >= 20:
            print("✅ Viable si hay suficientes datos por clase")
        else:
            print("⚠️  Algunas clases tienen muy pocas muestras")
    else:
        print(f"Demasiadas clases ({n_classes}) para clasificación exacta")
    
    print()
    
    # Análisis de tamaño de clusters
    if 'n_surface_atoms' in df.columns:
        print("="*70)
        print("🔬 ANÁLISIS DE TAMAÑO DE CLUSTERS")
        print("="*70)
        print()
        
        n_atoms = df['n_surface_atoms']
        
        print(f"Átomos superficiales por cluster:")
        print(f"  Mínimo:  {n_atoms.min():.0f}")
        print(f"  Máximo:  {n_atoms.max():.0f}")
        print(f"  Media:   {n_atoms.mean():.1f}")
        print(f"  Mediana: {n_atoms.median():.0f}")
        print(f"  Q25:     {n_atoms.quantile(0.25):.0f}")
        print(f"  Q75:     {n_atoms.quantile(0.75):.0f}")
        print()
        
        # Recomendar n_sample
        median_atoms = n_atoms.median()
        
        if median_atoms < 100:
            n_sample_rec = 128
            msg = "Clusters muy pequeños"
        elif median_atoms < 200:
            n_sample_rec = 256
            msg = "Clusters pequeños"
        elif median_atoms < 400:
            n_sample_rec = 512
            msg = "Clusters medianos"
        else:
            n_sample_rec = 1024
            msg = "Clusters grandes"
        
        print(f"📊 {msg}")
        print(f"   Recomendación: --n_sample {n_sample_rec}")
        print()
    
    # Recomendación final
    print("="*70)
    print("💡 RECOMENDACIÓN FINAL")
    print("="*70)
    print()
    
    print("Para tu dataset, la mejor estrategia es:")
    print()
    print("🎯 OPCIÓN 2: Clasificación 4 Clases")
    print("   - Single vacancy (1-2)")
    print("   - Small cluster (3-5)")
    print("   - Medium cluster (6-9)")
    print("   - Large cluster (10+)")
    print()
    print("Comando sugerido:")
    print()
    print(f"python generate_pointnet_data.py \\")
    print(f"    -i \"databases/db_integrate/*\" \\")
    print(f"    -c {Path(csv_path).name} \\")
    print(f"    -o data_pointnet \\")
    if 'n_surface_atoms' in df.columns:
        print(f"    -n {n_sample_rec} \\")
    else:
        print(f"    -n 256 \\")
    print(f"    --split")
    print()
    
    return df


def plot_distribution(df, output_file=None):
    """Genera gráficos de distribución"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de n_vacancies
    ax1 = axes[0]
    n_vac = df['n_vacancies']
    ax1.hist(n_vac, bins=range(int(n_vac.min()), int(n_vac.max())+2), 
             alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Número de Vacancias')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Vacancias')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot de tamaño de clusters
    if 'n_surface_atoms' in df.columns:
        ax2 = axes[1]
        
        # Agrupar por clases
        df['class'] = pd.cut(df['n_vacancies'], 
                            bins=[0, 2, 5, 9, float('inf')],
                            labels=['Single\n(1-2)', 'Small\n(3-5)', 
                                   'Medium\n(6-9)', 'Large\n(10+)'])
        
        df.boxplot(column='n_surface_atoms', by='class', ax=ax2)
        ax2.set_xlabel('Clase')
        ax2.set_ylabel('Átomos Superficiales')
        ax2.set_title('Tamaño de Clusters por Clase')
        plt.sca(ax2)
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfico guardado: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analiza dataset de vacancias y sugiere estrategia'
    )
    parser.add_argument('csv', help='Archivo CSV del dataset')
    parser.add_argument('--plot', action='store_true',
                       help='Generar gráficos')
    parser.add_argument('--output', help='Guardar gráfico en archivo')
    
    args = parser.parse_args()
    
    df = analyze_dataset(args.csv)
    
    if args.plot or args.output:
        plot_distribution(df, args.output)


if __name__ == '__main__':
    main()
