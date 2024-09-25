import numpy as np
import matplotlib.pyplot as plt

# Definir la función de densidad de probabilidad de la distribución normal
def normal_dist(x, mu, sigma):
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# Parámetros
mu = 0  # Media
sigma_1 = 0.5  # Primera desviación estándar
c = 0.02  # Segunda desviación estándar

x = np.linspace(-5, 5, 1000)  # Valores de x

# Calcular la función de densidad para cada desviación estándar
y_1 = normal_dist(x, mu, sigma_1/c)
y_2 = normal_dist(x, mu, sigma_1)


# Mostrar la gráfica
plt.show()

# Crear subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histograma para noise_1
axes[0].plot(x, y_2/c, label=f'$\sigma={sigma_1/c}$')
axes[0].set_title('Histograma del ruido 1')
axes[0].set_xlabel('Valor')
axes[0].set_ylabel('Frecuencia')

# Histograma para noise_2
axes[1].plot(x, y_1, label=f'$\sigma={sigma_1}$')
axes[1].set_title('Histograma del ruido 2')
axes[1].set_xlabel('Valor')
axes[1].set_ylabel('Frecuencia')

# Ajustar el espacio entre subplots
plt.tight_layout()

# Mostrar los histogramas en subplots
plt.show()