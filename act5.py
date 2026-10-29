import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ===========
# Etapa 1: Definicion del problema
# ===========
# Problema: predecir el rendimiento de combustible en millas por galon 
# segun la potencia HP de un automovil

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
df = pd.read_csv(url)

print("\m== PRIMEROS 5 REGISTROS ==")
print(df.head())

print("\n== DIMENSIONES Y COLUMNAS ==")
print("Dimensiones: ", df.shape)
print("Columnas: ", df.columns.tolist())

print("\n== INFO GENERAL Y TIPOS DE DATOS ==")
df.info()

print("\n== VALORES NULOS POR COLUMNA ==")
print(df.isnull().sum())

print("\n== REGISTROS DUPLICADOS ==")
print(f"Total de duplicados: {df.duplicated().sum()}")

print("\n== LIMPIANDO DATOS ==")
df = df.drop_duplicates()

df = df.dropna(subset=['horsepower', 'mpg'])

print("Dimensiones despues de la limpieza: ", df.shape)

#variables x (indep) y (depen)

x = df[['horsepower']]

y = df['mpg']


modelo = LinearRegression()
modelo.fit(x, y)

pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

print("\n== PARAMETROS DEL MODELO ==")
print(f"Pendiente (m): {pendiente:.4f}")
print(f"Intercepto (b): {intercepto:.4f}")
print(f"Ecuacion: MPG = ({pendiente:.4f} * Horsepower) + {intercepto:.4f}")

valores_hp = [75, 100, 125, 150, 175]
df_predicciones = pd.DataFrame({'horsepower': valores_hp})

df_predicciones['mpg_predicho'] = modelo.predict(df_predicciones[['horsepower']])

print("\n== PREDICCIONES DE RENDIMIENTO DE COMBUSTIBLE ==")
for _, fila in df_predicciones.iterrows():
    print(f"- {int(fila['horsepower'])} HP -> {fila['mpg_predicho']:.2f} MPG")
    
plt.figure(figsize=(10, 6))

plt.scatter(df['horsepower'], df['mpg'], color='steelblue', alpha=0.5, label='Datos reales')

y_pred_linea = modelo.predict(x)
plt.plot(df['horsepower'], y_pred_linea, color='red', linewidth=2, 
         label=f'Regresion: MPG = {pendiente:.2f}*HP + {intercepto:.2f}')

plt.scatter(df_predicciones['horsepower'], df_predicciones['mpg_predicho'], 
            color='darkgreen', s=90, zorder=5, label='Predicciones (75 a 175 HP)')

for _, fila in df_predicciones.iterrows():
    plt.annotate(f"{fila['mpg_predicho']:.1f}", 
                 (fila['horsepower'], fila['mpg_predicho']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='darkgreen', weight='bold')

plt.title('Relacion entre Potencia (HP) y Rendimiento de Combustible (MPG)', fontsize=13)
plt.xlabel('Potencia del Motor (Horsepower - HP)', fontsize=11)
plt.ylabel('Rendimiento (Millas por Galon - MPG)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()