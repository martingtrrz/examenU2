import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = {
    "publicidad": [1000, 2000, 3000, 4000, 5000],
    "ventas": [5000,7000,9000,11000,13000]
}

df = pd.DataFrame(datos)

print("Datos de la tienda: ")
print(df)

x = df[["publicidad"]]
y = df["ventas"]

modelo = LinearRegression()
modelo.fit(x, y)

print("\nPendiente: ", round(modelo.coef_[0], 2))
print("Intercepto: ", round(modelo.intercept_, 2))

#6. realizar una prediccion
nuevo_dato = pd.DataFrame({"publicidad": [6000]})
prediccion= modelo.predict(nuevo_dato)

print("Ventas estimadas: ", round(prediccion[0], 2))

plt.scatter(x["publicidad"], y, label="Datos reales")
plt.plot(x["publicidad"], modelo.predict(x),
         label="Regresion lineal")

plt.scatter(6000, prediccion[0],
            label="Prediccion", marker="x", s=100)

plt.xlabel("Inversion en publicidad ($)")
plt.ylabel("Ventas ($)")
plt.title("Regresion lineal simple")
plt.legend()
plt.grid(True)
plt.show()