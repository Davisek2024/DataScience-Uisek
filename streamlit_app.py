import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(layout='wide')

st.title('📊 Dashboard de Predicción de Deserción Estudiantil')
st.markdown('Modelo: **Random Forest** | Enfoque: **Variables Clave**')

# --- 1. Carga y Preprocesamiento de Datos ---
@st.cache_data
def load_data():
    # MODIFICADO: Usar ruta relativa para ejecución local
    df = pd.read_csv('uisek_desercion_CLEAN.csv')
    X = df.drop('deserto_semestre', axis=1)
    y = df['deserto_semestre']
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X, y, df

X, y, raw_df = load_data()

# --- 2. Entrenamiento del Modelo Random Forest ---
@st.cache_resource
def train_model(X_data, y_data):
    # Dividir los datos en conjuntos de entrenamiento y prueba (necesario para el entrenamiento)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42, stratify=y_data)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# --- 3. Obtener Importancia de Características ---
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# --- Sidebar para Input del Usuario ---
st.sidebar.header('Ajusta los Parámetros del Estudiante')

# Obtener valores medios/modales para prellenar
default_values = X.mean()
default_categorical_values = {
    'carrera_Derecho': 0, 'carrera_Diseño_Digital': 0,
    'carrera_Gastronomía': 0, 'carrera_Informática': 0,
    'carrera_Negocios_Digitales': 0, 'carrera_Software': 0
}
for col in default_categorical_values:
    if col in X.columns:
        default_categorical_values[col] = X[col].mode()[0] # Usar la moda para binarias

# Input para las variables más importantes
semestre_actual = st.sidebar.slider(
    'Semestre Actual', min_value=X['semestre_actual'].min(), max_value=X['semestre_actual'].max(),
    value=int(default_values['semestre_actual'])
)
promedio_notas = st.sidebar.slider(
    'Promedio de Notas', min_value=float(X['promedio_notas'].min()), max_value=float(X['promedio_notas'].max()),
    value=float(default_values['promedio_notas']),
    step=0.01
)
asistencia_porcentaje = st.sidebar.slider(
    'Porcentaje de Asistencia', min_value=float(X['asistencia_porcentaje'].min()), max_value=float(X['asistencia_porcentaje'].max()),
    value=float(default_values['asistencia_porcentaje']),
    step=0.1
)
deuda_pendiente_usd = st.sidebar.number_input(
    'Deuda Pendiente (USD)', min_value=float(X['deuda_pendiente_usd'].min()), max_value=float(X['deuda_pendiente_usd'].max()),
    value=float(default_values['deuda_pendiente_usd']),
    step=10.0
)
actividad_canvas_semanal = st.sidebar.number_input(
    'Actividad Canvas Semanal (horas)', min_value=float(X['actividad_canvas_semanal'].min()), max_value=float(X['actividad_canvas_semanal'].max()),
    value=float(default_values['actividad_canvas_semanal']),
    step=1.0
)
horas_trabajo_semanal = st.sidebar.number_input(
    'Horas de Trabajo Semanal', min_value=float(X['horas_trabajo_semanal'].min()), max_value=float(X['horas_trabajo_semanal'].max()),
    value=float(default_values['horas_trabajo_semanal']),
    step=1.0
)
materias_reprobadas_acum = st.sidebar.number_input(
    'Materias Reprobadas Acumuladas', min_value=int(X['materias_reprobadas_acum'].min()), max_value=int(X['materias_reprobadas_acum'].max()),
    value=int(default_values['materias_reprobadas_acum']),
    step=1
)
visitas_tutoria_semestre = st.sidebar.number_input(
    'Visitas a Tutoría Semestre', min_value=int(X['visitas_tutoria_semestre'].min()), max_value=int(X['visitas_tutoria_semestre'].max()),
    value=int(default_values['visitas_tutoria_semestre']),
    step=1
)

trabaja = st.sidebar.selectbox('¿Trabaja?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=int(default_values['trabaja']))
tiene_beca = st.sidebar.selectbox('¿Tiene Beca?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=int(default_values['tiene_beca']))
vive_fuera_de_quito = st.sidebar.selectbox('¿Vive fuera de Quito?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=int(default_values['vive_fuera_de_quito']))
participa_extracurricular = st.sidebar.selectbox('¿Participa en Extracurricular?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=int(default_values['participa_extracurricular']))
flag_cero_visitas_tutoria_semestre = st.sidebar.selectbox('¿Cero visitas a tutoría?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=int(default_values['flag_cero_visitas_tutoria_semestre']))
flag_cero_asistencia_porcentaje = st.sidebar.selectbox('¿Cero porcentaje de asistencia?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=int(default_values['flag_cero_asistencia_porcentaje']))

# Input para la columna 'carrera' (categórica)
carreras_disponibles = raw_df['carrera'].unique().tolist()
selected_carrera = st.sidebar.selectbox('Carrera', options=carreras_disponibles)


# --- Creación del DataFrame de Input para la Predicción ---
input_data = pd.DataFrame(columns=X.columns)
input_data.loc[0] = 0 # Inicializar con ceros

input_data['semestre_actual'] = semestre_actual
input_data['promedio_notas'] = promedio_notas
input_data['asistencia_porcentaje'] = asistencia_porcentaje
input_data['materias_reprobadas_acum'] = materias_reprobadas_acum
input_data['trabaja'] = trabaja
input_data['horas_trabajo_semanal'] = horas_trabajo_semanal
input_data['actividad_canvas_semanal'] = actividad_canvas_semanal
input_data['tiene_beca'] = tiene_beca
input_data['deuda_pendiente_usd'] = deuda_pendiente_usd
input_data['vive_fuera_de_quito'] = vive_fuera_de_quito
input_data['visitas_tutoria_semestre'] = visitas_tutoria_semestre
input_data['participa_extracurricular'] = participa_extracurricular
input_data['flag_cero_visitas_tutoria_semestre'] = flag_cero_visitas_tutoria_semestre
input_data['flag_cero_asistencia_porcentaje'] = flag_cero_asistencia_porcentaje

# Asignar la carrera seleccionada (One-Hot Encoded)
carrera_col_name = f'carrera_{selected_carrera}'
if carrera_col_name in input_data.columns:
    input_data[carrera_col_name] = 1


# --- Predicción ---
original_prediction_proba = model.predict_proba(input_data)[0][1]
display_prediction_proba = original_prediction_proba

st.subheader('Resultado de la Predicción')
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Probabilidad de Deserción", value=f"{display_prediction_proba*100:.2f}%")
with col2:
    if display_prediction_proba >= 0.30: # Usando el umbral de 0.30 como recomendado
        st.error('RIESGO DE DESERCIÓN')
        st.write('Se recomienda una intervención inmediata.')
    else:
        st.success('RIESGO BAJO DE DESERCIÓN')
        st.write('Monitoreo regular recomendado.')

st.markdown('---')

# --- Visualización de Importancia de Características ---
st.subheader('Importancia de las Características del Modelo')
st.write("Las variables se muestran ordenadas de mayor a menor importancia en la predicción.")
st.dataframe(importance_df)

st.markdown('---')
st.info("Ajusta los parámetros en la barra lateral izquierda para ver cómo afectan la probabilidad de deserción.")
