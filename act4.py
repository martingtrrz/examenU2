import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Cambio para la rama dev


datos = {
    "HorasEstudio": [1, 2, 3, 4, 5, 6, 7, 8],
    "Calificacion": [55, 60, 65, 70, 75, 80, 85, 90],
}

df = pd.DataFrame(datos)

x = df[["HorasEstudio"]]
y = df["Calificacion"]


modelo = LinearRegression()
modelo.fit(x, y)


pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

print(f"Pendiente (m): {pendiente}")
print(f"Intercepto (b): {intercepto}")
print(f"Ecuacion de la recta: Calificacion = {pendiente} * Horas + {intercepto}")


horas_a_predecir = np.array([[9]])
prediccion_9h = modelo.predict(horas_a_predecir)

print( f"\nPrediccion para 9 hroas de estudio: {prediccion_9h[0]:.2} puntos de calificacion")

linea_prediccion = modelo.predict(x)
plt.plot (
    df["HorasEstudio"],
    linea_prediccion,
    color="red",
    linewidth=2,
    label=f"Linea de regresion: y = {pendiente:.0f}x + {intercepto:.0}",
)

plt.scatter(
    [9],
    prediccion_9h,
    color="green",
    s=100,
    zorder=5,
    label=f"Prediccion (9h, {prediccion_9h[0]:.0f})",
)

plt.title("Relacion entre horas de estudio y calificacion obtenida")
plt.xlabel ("Horas de Estudio")
plt.ylabel("Calificacion")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()