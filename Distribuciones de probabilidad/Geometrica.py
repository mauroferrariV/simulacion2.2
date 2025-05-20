import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# Parámetros de la distribución geométrica
p = 0.3  # Probabilidad de éxito en cada ensayo
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución geométrica
datos = np.random.geometric(p, num_muestras)

# Configurar el histograma (bins centrados en enteros)
max_valor = datos.max()
bins = np.arange(0.5, max_valor + 1.5, 1)  # Bins alineados con enteros

plt.hist(datos, bins=bins, density=True, alpha=0.6, color='g', 
         edgecolor='black', label='Muestras')

# Calcular y graficar la PMF teórica
x = np.arange(1, max_valor + 1)  # Valores posibles (k ≥ 1)
pmf_teorica = geom.pmf(x, p)

plt.plot(x, pmf_teorica, 'ro--', linewidth=1.5, markersize=6, 
         label='PMF teórica')

# Personalización del gráfico
plt.title(f"Distribución Geométrica\np = {p}")
plt.xlabel('Número de ensayos hasta el primer éxito')
plt.ylabel('Densidad de probabilidad')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(np.arange(1, max_valor + 1))

plt.show()