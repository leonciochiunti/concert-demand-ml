# Concert Demand ML / ConcertDemandAI

## Descripción general

ConcertDemandAI es un MVP académico desarrollado para apoyar la planeación y evaluación de conciertos mediante técnicas de Inteligencia Artificial y Machine Learning.

El sistema permite estimar la demanda esperada de un concierto antes de su realización, clasificándola como baja, media o alta. Además, la aplicación muestra indicadores estimados del evento, recomendaciones de conciertos similares y un asistente básico de interpretación de resultados.

La propuesta está orientada principalmente a empresas organizadoras, promotores de conciertos, equipos de marketing y administradores de recintos, ya que busca apoyar la toma de decisiones relacionadas con precio, capacidad, ciudad, tipo de recinto, presupuesto de marketing y riesgo comercial.

## Problema que busca resolver

En la organización de conciertos, muchas decisiones se toman con base en experiencia, intuición o análisis limitados. Esto puede provocar problemas como baja asistencia, elección incorrecta del recinto, precios poco adecuados, presupuestos de marketing mal distribuidos o una estimación imprecisa de la demanda.

Este proyecto busca ofrecer una herramienta inicial que permita evaluar escenarios de conciertos a partir de datos estructurados, generando una predicción de demanda y métricas de apoyo para la toma de decisiones.

## Objetivo del proyecto

El objetivo principal es desarrollar un MVP funcional que aplique técnicas de Machine Learning para predecir la demanda de conciertos y apoyar la evaluación previa de eventos.

De manera específica, el sistema permite:

- Predecir si la demanda esperada de un concierto será baja, media o alta.
- Evaluar indicadores estimados como ocupación, boletos vendidos, ingreso bruto potencial y riesgo comercial.
- Recomendar conciertos similares a partir de características del dataset.
- Interpretar los resultados mediante un asistente básico de procesamiento de lenguaje natural.
- Proporcionar una interfaz sencilla y accesible para usuarios no técnicos mediante Streamlit.

## Alcance del MVP

### Funcionalidades implementadas

El MVP incluye las siguientes funcionalidades:

- Carga y uso de un dataset simulado realista de conciertos.
- Análisis exploratorio de datos.
- Entrenamiento y comparación de modelos de clasificación.
- Selección del mejor modelo con base en métricas de evaluación.
- Predicción de demanda del evento como baja, media o alta.
- Interfaz web interactiva desarrollada con Streamlit.
- Carga dinámica de opciones desde `data/dataset.csv`.
- Detección automática del país según la ciudad seleccionada.
- Detección automática del género musical según el artista seleccionado.
- Evaluación estimada del evento mediante:
  - ocupación estimada
  - boletos estimados
  - ingreso bruto potencial
  - riesgo comercial
  - relación entre ingreso bruto estimado y presupuesto de marketing
- Recomendación básica de conciertos similares.
- Asistente básico de interpretación de resultados mediante detección de palabras clave.

### Funcionalidades fuera del alcance actual

Debido al alcance temporal del proyecto, algunos elementos se consideran extensiones futuras:

- Integración con plataformas reales de venta de boletos.
- Uso de datos reales en tiempo real.
- Base de datos productiva.
- API distribuida.
- Precios dinámicos.
- Chatbot avanzado con modelos conversacionales externos.
- Sistema de recomendación personalizado con historial real de usuarios.

## Público objetivo

El sistema está orientado principalmente a:

- Empresas organizadoras de conciertos.
- Promotores de eventos.
- Equipos de marketing.
- Administradores de recintos.
- Analistas de demanda o planeación comercial.

En su versión actual, la aplicación funciona como una herramienta de apoyo para evaluar eventos antes de su realización. En futuras versiones, también podría ampliarse para usuarios finales mediante recomendaciones personalizadas de conciertos.

## Stack tecnológico

El proyecto utiliza herramientas de código abierto y uso gratuito:

- Python como lenguaje principal.
- Pandas y NumPy para procesamiento de datos.
- Scikit-learn para entrenamiento y evaluación de modelos de Machine Learning.
- Streamlit para la construcción de la interfaz interactiva.
- Joblib para guardar y cargar el modelo entrenado.
- Matplotlib para visualización en el análisis exploratorio.

## Estructura del proyecto

```bash
concert-demand-ml/
│
├── app/
│   └── app.py
│
├── data/
│   └── dataset.csv
│
├── model/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── model.pkl
│   ├── metrics.txt
│   └── metrics.json
│
├── notebooks/
│   ├── 01_eda_training.ipynb
│   ├── 02_train_model.ipynb
│   └── 03_integracion_modelo_app.ipynb
│
├── requirements.txt
└── README.md