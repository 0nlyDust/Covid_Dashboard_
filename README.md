# COVID-19 Dashboard Interactivo

Este proyecto es un **dashboard interactivo de COVID-19** desarrollado en Python utilizando **Dash**, **Plotly**, **pandas** y **scikit-learn**.

---

## Funcionalidades

- Casos diarios y acumulados por país
- Fallecimientos y recuperaciones (con eje secundario)
- Heatmap de correlaciones entre series
- Predicción lineal de casos diarios
- Selección de país y rango de fechas
- Estadísticas rápidas: máximo, mínimo y promedio de casos diarios
- Gráficas interactivas con hover, zoom y selección de rango de fechas

---

## Origen de los datos

Los datos utilizados provienen del **dataset gratuito de Josh** sobre COVID-19 ([CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19)), que contiene información de:

- Casos confirmados
- Fallecimientos
- Recuperaciones

> Gracias a este dataset gratuito, podemos generar dashboards científicos y visualizaciones interactivas para aprendizaje y portafolio.

---

## Cómo ejecutar el dashboard

1. Clona el repositorio:

git clone https://github.com/USERNAME/covid-dashboard-python.git
cd covid-dashboard-python

2. Instala las dependencias:

pip install -r requirements.txt

3. Ejecuta la app:

python covid_dashboard.py

4. Abre tu navegador y visita:

http://127.0.0.1:8050
