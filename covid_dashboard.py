import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                            marks={i:str(date.date()) for i,date in enumerate(sorted(df_pais_total['Fecha'].unique()))},
                            step=1)
        ], style={'width':'48%', 'display':'inline-block', 'padding-left':'20px'})
    ], style={'margin-bottom':'30px'}),

    # Gráficas
    html.Div([
        dcc.Graph(id='grafico-casos'),
        dcc.Graph(id='heatmap-correlacion'),
        dcc.Graph(id='grafico-prediccion')
    ])
])

@app.callback(
    [Output('grafico-casos','figure'),
     Output('heatmap-correlacion','figure'),
     Output('grafico-prediccion','figure')],
    [Input('dropdown-pais','value'),
     Input('slider-fechas','value')]
)
def actualizar_dashboard(pais_seleccionado, rango_fechas):
    df_sel = df_pais_total[df_pais_total['Country/Region']==pais_seleccionado].sort_values('Fecha').reset_index(drop=True)
    df_sel = df_sel.iloc[rango_fechas[0]:rango_fechas[1]+1].copy()

    df_sel['Casos_diarios'] = df_sel['Casos_acumulados'].diff().fillna(df_sel['Casos_acumulados'])
    df_sel['Fallecimientos_diarios'] = df_sel['Fallecimientos'].diff().fillna(df_sel['Fallecimientos'])
    df_sel['Recuperaciones_diarias'] = df_sel['Recuperaciones'].diff().fillna(df_sel['Recuperaciones'])

    # Verificación de valores NaN
    print(df_sel[['Fecha', 'Fallecimientos', 'Fallecimientos_diarios']].isna().sum())

    # Gráfico evolución
    fig_casos = go.Figure()

    fig_casos.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Casos_diarios'], mode='lines+markers', name='Casos diarios'))
    fig_casos.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Casos_acumulados'], mode='lines+markers', name='Casos acumulados'))
    fig_casos.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Fallecimientos_diarios'], mode='lines+markers', name='Fallecimientos diarios', yaxis="y2"))
    fig_casos.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Recuperaciones_diarias'], mode='lines+markers', name='Recuperaciones diarias'))

    fig_casos.update_layout(
        title=f"Evolución COVID-19 en {pais_seleccionado}",
        xaxis_title="Fecha",
        yaxis_title="Número de casos",
        yaxis2=dict(title="Fallecimientos diarios", overlaying="y", side="right")
    )

    # Heatmap correlación
    corr = df_sel[['Casos_diarios','Fallecimientos_diarios','Recuperaciones_diarias']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title=f"Correlaciones COVID-19 en {pais_seleccionado}")

    # Predicción
    df_sel['Fecha_ordinal'] = df_sel['Fecha'].map(pd.Timestamp.toordinal)
    X = df_sel[['Fecha_ordinal']]
    y = df_sel['Casos_diarios']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df_sel['Fecha'], y=df_sel['Casos_diarios'], mode='lines+markers', name='Casos reales'))
    fig_pred.add_trace(go.Scatter(x=df_sel['Fecha'].iloc[X_train.shape[0]:], y=y_pred, mode='lines+markers', name='Predicción'))
    fig_pred.update_layout(title=f"Predicción de casos diarios en {pais_seleccionado}", xaxis_title="Fecha", yaxis_title="Casos diarios")

    return fig_casos, fig_corr, fig_pred

if __name__ == '__main__':

    app.run(debug=True)
