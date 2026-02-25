import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# ---------- Datos ----------
def transformar_csv(ruta, valor_columna):
    df = pd.read_csv(ruta)
    df_long = df.melt(id_vars=['Province/State','Country/Region','Lat','Long'],
                      var_name='Fecha', value_name=valor_columna)
    df_long['Fecha'] = pd.to_datetime(df_long['Fecha'], format='%m/%d/%y')
    return df_long

ruta_confirmed = r"C:\Users\maria\Desktop\COVID\COVID-19-master\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv"
ruta_deaths = r"C:\Users\maria\Desktop\COVID\COVID-19-master\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_deaths_global.csv"
ruta_recovered = r"C:\Users\maria\Desktop\COVID\COVID-19-master\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_recovered_global.csv"

df_confirmed = transformar_csv(ruta_confirmed, 'Casos_acumulados')
df_deaths = transformar_csv(ruta_deaths, 'Fallecimientos')
df_recovered = transformar_csv(ruta_recovered, 'Recuperaciones')

df_merged = df_confirmed.merge(df_deaths[['Country/Region','Province/State','Fecha','Fallecimientos']],
                               on=['Country/Region','Province/State','Fecha'])
df_merged = df_merged.merge(df_recovered[['Country/Region','Province/State','Fecha','Recuperaciones']],
                            on=['Country/Region','Province/State','Fecha'])

df_pais_total = df_merged.groupby(['Country/Region','Fecha']).sum().reset_index()

# ---------- Dash ----------
app = Dash(__name__)
paises = df_pais_total['Country/Region'].unique()

app.layout = html.Div([
    html.H1("Dashboard COVID-19", style={'text-align':'center', 'margin-bottom':'20px'}),

    # Controles arriba
    html.Div([
        html.Div([
            html.Label("Selecciona un país:"),
            dcc.Dropdown(id='dropdown-pais',
                         options=[{'label': p, 'value': p} for p in paises],
                         value='Spain',
                         clearable=False,
                         searchable=False)
        ], style={'width':'48%', 'display':'inline-block'}),

        html.Div([
            html.Label("Selecciona rango de fechas:"),
            dcc.RangeSlider(id='slider-fechas',
                            min=0,
                            max=df_pais_total['Fecha'].nunique()-1,
                            value=[0, df_pais_total['Fecha'].nunique()-1],
                            marks={i:f"Día {i+1}" for i in range(df_pais_total['Fecha'].nunique())},
                            step=1)
        ], style={'width':'48%', 'display':'inline-block', 'padding-left':'20px'}),
        
        html.Div([
            html.Label("Escala logarítmica:"),
            dcc.Checklist(id='log-scale',
                          options=[{'label': 'Usar escala logarítmica', 'value': 'log'}],
                          value=[],
                          style={'padding-top': '10px'})
        ], style={'width':'48%', 'display':'inline-block', 'padding-left':'20px'})
    ], style={'margin-bottom':'30px'}),

    # Gráficas
    html.Div([
        dcc.Graph(id='grafico-casos-diarios'),
        dcc.Graph(id='grafico-acumulados'),
        dcc.Graph(id='grafico-fallecimientos-diarios'),
        dcc.Graph(id='heatmap-correlacion'),
        dcc.Graph(id='grafico-prediccion')
    ])
])

@app.callback(
    [Output('grafico-casos-diarios', 'figure'),
     Output('grafico-acumulados', 'figure'),
     Output('grafico-fallecimientos-diarios', 'figure'),
     Output('heatmap-correlacion', 'figure'),
     Output('grafico-prediccion', 'figure')],
    [Input('dropdown-pais', 'value'),
     Input('slider-fechas', 'value'),
     Input('log-scale', 'value')]
)
def actualizar_dashboard(pais_seleccionado, rango_fechas, log_scale):
    df_sel = df_pais_total[df_pais_total['Country/Region'] == pais_seleccionado].sort_values('Fecha').reset_index(drop=True)
    df_sel = df_sel.iloc[rango_fechas[0]:rango_fechas[1] + 1].copy()

    # Calcular casos y fallecimientos diarios
    df_sel['Casos_diarios'] = df_sel['Casos_acumulados'].diff().fillna(df_sel['Casos_acumulados'])
    df_sel['Fallecimientos_diarios'] = df_sel['Fallecimientos'].diff().fillna(df_sel['Fallecimientos'])
    df_sel['Recuperaciones_diarias'] = df_sel['Recuperaciones'].diff().fillna(df_sel['Recuperaciones'])

    # Corregir valores negativos
    df_sel['Fallecimientos_diarios'] = df_sel['Fallecimientos_diarios'].apply(lambda x: max(0, x))

    # Media móvil de 7 días
    df_sel['Casos_diarios_movil'] = df_sel['Casos_diarios'].rolling(window=7).mean()
    df_sel['Fallecimientos_diarios_movil'] = df_sel['Fallecimientos_diarios'].rolling(window=7).mean()

    # Configurar escala logarítmica
    scale = 'log' if 'log' in log_scale else 'linear'

    # Gráfico de casos diarios
    fig_casos = go.Figure()
    fig_casos.add_trace(go.Bar(x=df_sel['Fecha'], y=df_sel['Casos_diarios'], name='Casos diarios', marker_color='royalblue'))
    fig_casos.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Casos_diarios_movil'], name='Media móvil 7 días', mode='lines', line=dict(color='red')))
    fig_casos.update_layout(
        title=f"Casos Diarios de COVID-19 en {pais_seleccionado}",
        xaxis_title="Fecha",
        yaxis_title="Casos diarios",
        xaxis=dict(tickformat="%d-%m-%Y"),
        yaxis=dict(type=scale),
        template="plotly_dark"
    )

    # Gráfico de casos acumulados
    fig_acumulados = go.Figure()
    fig_acumulados.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Casos_acumulados'], mode='lines', name='Casos acumulados', line=dict(color='green')))
    fig_acumulados.update_layout(
        title=f"Casos Acumulados de COVID-19 en {pais_seleccionado}",
        xaxis_title="Fecha",
        yaxis_title="Casos acumulados",
        xaxis=dict(tickformat="%d-%m-%Y"),
        yaxis=dict(type=scale),
        template="plotly_dark"
    )

    # Gráfico de fallecimientos diarios
    fig_fallecimientos = go.Figure()
    fig_fallecimientos.add_trace(go.Bar(x=df_sel['Fecha'], y=df_sel['Fallecimientos_diarios'], name='Fallecimientos diarios', marker_color='darkred'))
    fig_fallecimientos.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Fallecimientos_diarios_movil'], name='Media móvil 7 días', mode='lines', line=dict(color='orange')))
    fig_fallecimientos.update_layout(
        title=f"Fallecimientos Diarios por COVID-19 en {pais_seleccionado}",
        xaxis_title="Fecha",
        yaxis_title="Fallecimientos diarios",
        xaxis=dict(tickformat="%d-%m-%Y"),
        yaxis=dict(type=scale),
        template="plotly_dark"
    )

    # Heatmap de correlaciones
    corr = df_sel[['Casos_diarios', 'Fallecimientos_diarios', 'Recuperaciones_diarias']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title=f"Correlaciones COVID-19 en {pais_seleccionado}")
    fig_corr.update_layout(
        template="plotly_dark"
    )

    # Predicción de casos con regresión lineal para todo el conjunto de datos
    df_sel['Fecha_ordinal'] = df_sel['Fecha'].map(pd.Timestamp.toordinal)
    X = df_sel[['Fecha_ordinal']]
    y = df_sel['Casos_diarios']
    
    # Modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicción para todo el conjunto de fechas
    y_pred = model.predict(X)
    
    # Suavizado con media móvil
    y_pred_smooth = pd.Series(y_pred).rolling(window=7).mean()

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Casos_diarios'], mode='lines', name='Casos reales', line=dict(color='blue')))
    fig_pred.add_trace(go.Scatter(x=df_sel['Fecha'], y=y_pred_smooth, mode='lines', name='Predicción suavizada', line=dict(color='orange')))
    fig_pred.update_layout(
        title=f"Predicción de Casos Diarios en {pais_seleccionado}",
        xaxis_title="Fecha",
        yaxis_title="Casos diarios",
        xaxis=dict(tickformat="%d-%m-%Y"),
        yaxis=dict(type=scale),
        template="plotly_dark"
    )

    return fig_casos, fig_acumulados, fig_fallecimientos, fig_corr, fig_pred

if __name__ == '__main__':
    app.run(debug=True)
