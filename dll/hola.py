import numpy as np
import matplotlib.pyplot as plt

# Parámetros
t = np.linspace(0, 10, 1000000)  # Vector de tiempo o valores independientes
c = 0.002
std_noise = 0.5  # Desviación estándar del ruido

# Generar ruido con distribución normal
noise_1 = np.random.normal(loc=0, scale=std_noise/c, size=len(t))

# Generar ruido con distribución normal
noise_2 = np.random.normal(loc=0, scale=std_noise, size=len(t))/c

# Crear subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histograma para noise_1
axes[0].hist(noise_1, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Histograma del ruido 1')
axes[0].set_xlabel('Valor')
axes[0].set_ylabel('Frecuencia')

# Histograma para noise_2
axes[1].hist(noise_2, bins=30, color='skyblue', edgecolor='black')
axes[1].set_title('Histograma del ruido 2')
axes[1].set_xlabel('Valor')
axes[1].set_ylabel('Frecuencia')

# Ajustar el espacio entre subplots
plt.tight_layout()

# Mostrar los histogramas en subplots
plt.show()
