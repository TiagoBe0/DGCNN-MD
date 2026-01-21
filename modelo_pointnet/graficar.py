import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
import json
import numpy as np

# Cargar datos del archivo
with open('history.txt', 'r') as f:
    history = json.load(f)

# Extraer los datos
train_loss = history['train_loss']
train_acc = history['train_acc']
test_loss = history['test_loss']
test_acc = history['test_acc']

# Crear array de épocas
epochs = range(1, len(train_loss) + 1)

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(15, 10))

# Gráfico 1: Pérdida (Loss)
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Entrenamiento')
plt.plot(epochs, test_loss, 'r-', linewidth=2, label='Validación')
plt.xlabel('Época', fontsize=12)
plt.ylabel('Pérdida', fontsize=12)
plt.title('Pérdida durante el entrenamiento', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
min_test_loss = min(test_loss)
min_epoch = test_loss.index(min_test_loss) + 1
plt.axvline(x=min_epoch, color='g', linestyle='--', alpha=0.5, 
            label=f'Mínimo: época {min_epoch}')
plt.legend()

# Gráfico 2: Precisión (Accuracy)
plt.subplot(2, 2, 2)
plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Entrenamiento')
plt.plot(epochs, test_acc, 'r-', linewidth=2, label='Validación')
plt.xlabel('Época', fontsize=12)
plt.ylabel('Precisión (%)', fontsize=12)
plt.title('Precisión durante el entrenamiento', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
max_test_acc = max(test_acc)
max_epoch = test_acc.index(max_test_acc) + 1
plt.axvline(x=max_epoch, color='g', linestyle='--', alpha=0.5, 
            label=f'Máximo: época {max_epoch}')
plt.legend()

# Gráfico 3: Pérdida (escala logarítmica)
plt.subplot(2, 2, 3)
plt.semilogy(epochs, train_loss, 'b-', linewidth=2, label='Entrenamiento')
plt.semilogy(epochs, test_loss, 'r-', linewidth=2, label='Validación')
plt.xlabel('Época', fontsize=12)
plt.ylabel('Pérdida (log)', fontsize=12)
plt.title('Pérdida en escala logarítmica', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, which='both')

# Gráfico 4: Resumen estadístico
plt.subplot(2, 2, 4)
plt.axis('off')

stats_text = f"""
ESTADÍSTICAS DEL ENTRENAMIENTO
{'='*35}

Duración: {len(epochs)} épocas

ENTRENAMIENTO:
- Pérdida final: {train_loss[-1]:.4f}
- Mejor pérdida: {min(train_loss):.4f}
- Precisión final: {train_acc[-1]:.2f}%
- Mejor precisión: {max(train_acc):.2f}%

VALIDACIÓN:
- Pérdida final: {test_loss[-1]:.4f}
- Mejor pérdida: {min_test_loss:.4f} (época {min_epoch})
- Precisión final: {test_acc[-1]:.2f}%
- Mejor precisión: {max_test_acc:.2f}% (época {max_epoch})

DIFERENCIA FINAL:
- Pérdida: {abs(train_loss[-1] - test_loss[-1]):.4f}
- Precisión: {abs(train_acc[-1] - test_acc[-1]):.2f}%
"""

plt.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.suptitle('Análisis del Entrenamiento PyTorch Geometric', 
             fontsize=16, fontweight='bold', y=1.02)

# Guardar la figura
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.savefig('training_history.pdf', bbox_inches='tight')
plt.close()  # Cerrar la figura para liberar memoria

print(f"✓ Gráficos guardados como 'training_history.png' y 'training_history.pdf'")
print(f"✓ Estadísticas calculadas para {len(epochs)} épocas")
print(f"✓ Mejor precisión en validación: {max_test_acc:.2f}% (época {max_epoch})")
print(f"✓ Mejor pérdida en validación: {min_test_loss:.4f} (época {min_epoch})")

# Gráfico adicional de zoom
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

zoom_start = int(len(epochs) * 0.75)
zoom_epochs = list(epochs)[zoom_start:]

ax1.plot(zoom_epochs, train_loss[zoom_start:], 'b-', linewidth=2, 
         label='Entrenamiento')
ax1.plot(zoom_epochs, test_loss[zoom_start:], 'r-', linewidth=2, 
         label='Validación')
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Pérdida', fontsize=12)
ax1.set_title(f'Pérdida (últimas {len(zoom_epochs)} épocas)', 
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(zoom_epochs, train_acc[zoom_start:], 'b-', linewidth=2, 
         label='Entrenamiento')
ax2.plot(zoom_epochs, test_acc[zoom_start:], 'r-', linewidth=2, 
         label='Validación')
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Precisión (%)', fontsize=12)
ax2.set_title(f'Precisión (últimas {len(zoom_epochs)} épocas)', 
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_zoom.png', dpi=300, bbox_inches='tight')
plt.close()  # Cerrar la figura

print("✓ Gráfico de zoom guardado como 'training_zoom.png'")
