#!/usr/bin/env python3
"""
Convertidor LAMMPS dump → OFF para PointNet++
Autor: Santiago - Análisis MD con PointNet++

Convierte archivos .dump de LAMMPS al formato .off que usa PointNet++
Ideal para entrenar modelos en estructuras atómicas
"""

import numpy as np
import re
from pathlib import Path
import argparse


class DumpToOFF:
    """Convertidor de archivos dump de LAMMPS a formato OFF"""
    
    def __init__(self, dump_file):
        self.dump_file = Path(dump_file)
        self.atoms = []
        self.atom_types = []
        self.box_bounds = None
        
    def read_dump(self, timestep=-1, atom_type_filter=None):
        """
        Lee archivo dump de LAMMPS
        
        Args:
            timestep: Timestep específico a leer (-1 = último)
            atom_type_filter: Lista de tipos de átomo a incluir (None = todos)
        """
        print(f"Leyendo {self.dump_file}...")
        
        with open(self.dump_file, 'r') as f:
            lines = f.readlines()
        
        # Encontrar todos los timesteps
        timestep_indices = [i for i, line in enumerate(lines) 
                          if 'ITEM: TIMESTEP' in line]
        
        if not timestep_indices:
            raise ValueError("No se encontraron timesteps en el archivo")
        
        # Seleccionar timestep
        if timestep == -1:
            idx = timestep_indices[-1]  # Último timestep
            print(f"Usando último timestep (índice {len(timestep_indices)-1})")
        else:
            if timestep >= len(timestep_indices):
                raise ValueError(f"Timestep {timestep} no existe. Hay {len(timestep_indices)} timesteps")
            idx = timestep_indices[timestep]
            print(f"Usando timestep {timestep}")
        
        # Determinar el rango de líneas para este timestep
        if timestep_indices.index(idx) < len(timestep_indices) - 1:
            end_idx = timestep_indices[timestep_indices.index(idx) + 1]
        else:
            end_idx = len(lines)
        
        # Parsear datos
        i = idx
        while i < end_idx:
            line = lines[i].strip()
            
            if 'ITEM: NUMBER OF ATOMS' in line:
                n_atoms = int(lines[i+1].strip())
                print(f"Número de átomos: {n_atoms}")
                
            elif 'ITEM: BOX BOUNDS' in line:
                # Leer dimensiones de la caja
                self.box_bounds = []
                for j in range(3):
                    bounds = lines[i+1+j].strip().split()
                    self.box_bounds.append([float(bounds[0]), float(bounds[1])])
                print(f"Dimensiones caja: {self.box_bounds}")
                
            elif 'ITEM: ATOMS' in line:
                # Detectar formato de columnas
                header = line.replace('ITEM: ATOMS', '').strip().split()
                print(f"Columnas detectadas: {header[:10]}...")  # Mostrar solo primeras 10
                
                # Encontrar índices de columnas importantes
                try:
                    # Intentar encontrar columnas básicas
                    id_idx = header.index('id') if 'id' in header else 0
                    type_idx = header.index('type') if 'type' in header else 1
                    x_idx = header.index('x')
                    y_idx = header.index('y')
                    z_idx = header.index('z')
                    print(f"Usando columnas: x={x_idx}, y={y_idx}, z={z_idx}")
                except ValueError as e:
                    print(f"Error: Columnas x, y, z no encontradas")
                    print(f"Columnas disponibles: {header}")
                    raise
                
                # Leer átomos
                self.atoms = []
                self.atom_types = []
                
                for j in range(1, n_atoms + 1):
                    atom_data = lines[i+j].strip().split()
                    atom_type = int(atom_data[type_idx])
                    
                    # Filtrar por tipo si se especificó
                    if atom_type_filter is None or atom_type in atom_type_filter:
                        x = float(atom_data[x_idx])
                        y = float(atom_data[y_idx])
                        z = float(atom_data[z_idx])
                        
                        self.atoms.append([x, y, z])
                        self.atom_types.append(atom_type)
                
                print(f"Átomos leídos: {len(self.atoms)}")
                if atom_type_filter:
                    print(f"Filtrados por tipos: {atom_type_filter}")
                break
            
            i += 1
        
        self.atoms = np.array(self.atoms)
        self.atom_types = np.array(self.atom_types)
        
    def normalize_coordinates(self, center=True, scale=True):
        """
        Normaliza coordenadas atómicas
        
        Args:
            center: Centrar en el origen
            scale: Escalar a rango [-1, 1]
        """
        if len(self.atoms) == 0:
            return
        
        if center:
            centroid = np.mean(self.atoms, axis=0)
            self.atoms -= centroid
            print(f"Coordenadas centradas en: {centroid}")
        
        if scale:
            max_dist = np.max(np.abs(self.atoms))
            self.atoms /= max_dist
            print(f"Coordenadas escaladas por: {max_dist}")
    
    def write_off(self, output_file, n_sample=None):
        """
        Escribe archivo OFF
        
        Args:
            output_file: Ruta del archivo de salida
            n_sample: Número de puntos a muestrear (None = todos)
        """
        if len(self.atoms) == 0:
            raise ValueError("No hay átomos para escribir")
        
        # Muestrear si se especificó
        atoms_to_write = self.atoms
        if n_sample and n_sample < len(self.atoms):
            indices = np.random.choice(len(self.atoms), n_sample, replace=False)
            atoms_to_write = self.atoms[indices]
            print(f"Muestreados {n_sample} de {len(self.atoms)} átomos")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Header OFF
            f.write("OFF\n")
            f.write(f"{len(atoms_to_write)} 0 0\n")  # n_vertices, n_faces, n_edges
            
            # Coordenadas
            for atom in atoms_to_write:
                f.write(f"{atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n")
        
        print(f"✅ Archivo guardado: {output_file}")
        print(f"   Átomos escritos: {len(atoms_to_write)}")


def main():
    parser = argparse.ArgumentParser(
        description='Convertir archivos LAMMPS .dump a formato .off para PointNet++',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Convertir último timestep de un dump
  python dump_to_off.py input.dump output.off
  
  # Convertir timestep específico
  python dump_to_off.py input.dump output.off --timestep 0
  
  # Filtrar solo átomos de hierro (tipo 1) y normalizar
  python dump_to_off.py input.dump output.off --atom-types 1 --normalize
  
  # Muestrear 1024 puntos
  python dump_to_off.py input.dump output.off --n-sample 1024
        """
    )
    
    parser.add_argument('input', help='Archivo .dump de entrada')
    parser.add_argument('output', help='Archivo .off de salida')
    parser.add_argument('--timestep', type=int, default=-1,
                       help='Timestep a extraer (-1 = último)')
    parser.add_argument('--atom-types', type=int, nargs='+',
                       help='Tipos de átomo a incluir (ej: 1 2)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalizar coordenadas (centrar y escalar)')
    parser.add_argument('--n-sample', type=int,
                       help='Número de puntos a muestrear')
    
    args = parser.parse_args()
    
    # Crear convertidor
    converter = DumpToOFF(args.input)
    
    # Leer dump
    converter.read_dump(
        timestep=args.timestep,
        atom_type_filter=args.atom_types
    )
    
    # Normalizar si se pidió
    if args.normalize:
        converter.normalize_coordinates()
    
    # Escribir OFF
    converter.write_off(args.output, n_sample=args.n_sample)
    
    print("\n🎯 Conversión completada!")


if __name__ == '__main__':
    main()
