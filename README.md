# 🎤 Concert Demand ML

## Descripción
Este proyecto implementa un **MVP académico** para predecir la demanda de conciertos usando Machine Learning y mostrar el resultado en una interfaz sencilla con Streamlit.

## Objetivo
- Predecir la demanda de un concierto como **baja, media o alta**
- Mostrar una explicación básica del resultado
- Sugerir conciertos similares con una regla ligera de recomendación

## Alcance del MVP
### Incluye
- Dataset fake realista
- Entrenamiento de modelo de clasificación
- Evaluación básica
- Interfaz Streamlit
- Recomendación simple por género/ciudad

### No incluye
- Chatbot / NLP
- Integraciones externas
- Precios dinámicos en tiempo real
- API distribuida

## Stack
- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

## Estructura
```bash
concert-demand-ml/
│
├── data/
│   └── dataset.csv
├── model/
│   ├── train.py
│   ├── predict.py
│   └── model.pkl
├── notebooks/
│   └── 01_eda_training.ipynb
├── app/
│   └── app.py
├── notebooks/
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\\Scripts\\activate    # Windows

pip install -r requirements.txt
```

## Uso
### 1) Entrenar el modelo
```bash
python model/train.py
```

### 2) Ejecutar la app
```bash
streamlit run app/app.py
```

## Dataset
El dataset contiene:
- artista
- género
- ciudad
- pais
- tipo de venue
- capacidad
- mes
- día del evento
- días restantes
- presupuesto de marketing
- precio del boleto
- popularidad del artista
- ocupación estimada
- boletos vendidos
- clase de demanda (target)

## Modelo
Se usa un pipeline con:
- `OneHotEncoder` para variables categóricas
- `Logistic Regression` como modelo principal

## Equipo sugerido
- Persona 1: datos, features, entrenamiento, evaluación
- Persona 2: interfaz Streamlit, integración, demo

## Próximos pasos
- Comparar con Logistic Regression
- Agregar gráficos de distribución
- Mejorar explicaciones del modelo
- Guardar predicciones en SQLite

## Interpretación de resultados

La aplicación predice la demanda esperada de un concierto como baja, media o alta. Además, muestra una evaluación estimada del evento con indicadores como ocupación estimada, boletos estimados, ingreso estimado y riesgo comercial.

El ingreso estimado representa el ingreso bruto potencial por venta de boletos, calculado a partir de los boletos estimados y el precio del boleto. No representa utilidad neta, ya que no considera costos de producción, renta del recinto, pago al artista, personal, seguridad, impuestos, comisiones ni otros gastos operativos.
