import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Crear un DataFrame de ejemplo
data = {
    'Categorias': ['Superconducting', 'Trapped ions', 'Cold atoms', 'Quantum optics',
                   'Quantum spin \n in silicon', 'Quantum spin \n not in silicon',
                   'Topological', 'Other'],
    'Lead candidate': [17, 9, 3, 4, 2, 0, 0, 0],
    'Very promising': [14, 17, 15, 8, 9, 4, 0, 1],
    'Some potential': [5, 8, 14, 20, 21, 19, 13, 3],
    'Not promising': [0, 1, 3, 3, 2, 7, 17, 5],
    'No opinion': [0, 1, 1, 1, 2, 6, 6, 27]
}

df = pd.DataFrame(data)

# Configurar el DataFrame para tener las categorías como índices
df.set_index('Categorias', inplace=True)

# Crear un gráfico de barras apiladas
fig, ax = plt.subplots(figsize=(10, 6))
ax = df.plot(kind='bar', stacked=True, ax=ax, colormap='cividis', width=0.8, alpha=0.7)

# Añadir el valor de cada barra utilizando ax.text()
for i, (idx, row) in enumerate(df.iterrows()):
    for col in df.columns:
        value = row[col]
        if value != 0:
            ax.text(i, row.cumsum()[col] - value / 2, f'{value}', color='black',
                    ha='center', va='center', fontsize=8)

# Configurar el eje y para contar de 4 en 4 desde 0 hasta 36
plt.yticks(np.arange(0, 37, 4), fontname='Times New Roman')

# Personalizar la gráfica
ax.set_ylabel('Number of respondents', fontname='Times New Roman')
ax.set_xlabel('Technologies', fontname='Times New Roman')

# Configurar el título en un tamaño más grande y en negrita
ax.set_title('2022 Experts opinion on the potential of \n physical implementations for quantum computing',
             fontname='Times New Roman', fontsize=14, fontweight='bold')

# Configurar el tipo de letra del legend
ax.legend(title='Potential', bbox_to_anchor=(1, 1), prop={'family': 'Times New Roman'})

# Cambiar las etiquetas del eje x a Times New Roman y reducir el tamaño de letra
ax.set_xticklabels(ax.get_xticklabels(), fontname='Times New Roman', fontsize=8)

# Mostrar la gráfica
plt.show()
