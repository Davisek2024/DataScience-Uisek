import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = 'rf_desercion_uisek.joblib'
SCALER_FILE = 'scaler_uisek.joblib'
FEAT_FILE = 'feature_names_uisek.joblib'
CLEAN_DATA_FILE = 'uisek_desercion_CLEAN.csv'

# ---------- Cargar o generar artefactos exportados ----------
@st.cache_resource
def cargar_o_generar_artefactos():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEAT_FILE)):
        if not os.path.exists(CLEAN_DATA_FILE):
            st.error(f"No se encontró el archivo de datos '{CLEAN_DATA_FILE}'. "
                     "Asegúrate de ejecutar la app desde el directorio correcto.")
            st.stop()
        
        df = pd.read_csv(CLEAN_DATA_FILE)
        X = df.drop('deserto_semestre', axis=1)
        y = df['deserto_semestre']

        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        num_cols_to_scale = ['semestre_actual', 'promedio_notas', 'asistencia_porcentaje',
                             'materias_reprobadas_acum', 'horas_trabajo_semanal',
                             'actividad_canvas_semanal', 'deuda_pendiente_usd',
                             'visitas_tutoria_semestre']

        scaler = StandardScaler()
        scaler.fit(X_train[num_cols_to_scale])

        rand_forest = RandomForestClassifier(random_state=42)
        rand_forest.fit(X_train, y_train)

        joblib.dump(rand_forest, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(list(X.columns), FEAT_FILE)

    modelo = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feat = joblib.load(FEAT_FILE)
    return modelo, scaler, feat

modelo, scaler, feat = cargar_o_generar_artefactos()

# ---------- Configuración de página ----------
st.set_page_config(page_title='UISEK | Alerta de Deserción', layout='wide')
st.title('Sistema de Alerta Temprana de Deserción Estudiantil - UISEK')
st.caption('Modelo: Random Forest | Umbral: 0.30 | Artefactos cargados con joblib')

# ---------- Tabs ----------
tab1, tab2 = st.tabs(['Simular Estudiante', 'Scoring Batch (CSV)'])

# --- Tab 1: Simulador individual ---
with tab1:
    st.subheader('Simula el riesgo de deserción de un estudiante')

    st.sidebar.header('Datos del Estudiante')

    # Variables principales
    carrera = st.sidebar.selectbox('Carrera',
        ['Informatica', 'Negocios_Digitales', 'Software', 'TI'])
    semestre_actual = st.sidebar.slider('Semestre Actual', 1, 10, 4)
    promedio_notas = st.sidebar.slider('Promedio de Notas (0-10)', 0.0, 10.0, 7.0, 0.1)
    asistencia_porcentaje = st.sidebar.slider('% Asistencia', 0.0, 100.0, 75.0, 0.5)
    materias_reprobadas_acum = st.sidebar.slider('Materias Reprobadas Acumuladas', 0, 10, 1)

    # Variables financieras
    st.sidebar.markdown('---')
    st.sidebar.subheader('Información Financiera')
    tiene_beca = st.sidebar.selectbox('¿Tiene Beca?', [0, 1],
        format_func=lambda x: 'Sí' if x == 1 else 'No')
    deuda_pendiente_usd = st.sidebar.slider('Deuda Pendiente (USD)', 0.0, 3500.0, 500.0, 50.0)

    # Variables de comportamiento
    st.sidebar.markdown('---')
    st.sidebar.subheader('Comportamiento')
    trabaja = st.sidebar.selectbox('¿Trabaja?', [0, 1],
        format_func=lambda x: 'Sí' if x == 1 else 'No')
    horas_trabajo_semanal = st.sidebar.slider('Horas de Trabajo Semanal',
        0, 50, 0 if trabaja == 0 else 20)
    actividad_canvas_semanal = st.sidebar.slider(
        'Actividad Canvas Semanal (interacciones)', 0, 25, 12)
    vive_fuera_de_quito = st.sidebar.selectbox('¿Vive fuera de Quito?', [0, 1],
        format_func=lambda x: 'Sí' if x == 1 else 'No')
    visitas_tutoria_semestre = st.sidebar.slider('Visitas a Tutoría (semestre)', 0, 10, 2)
    participa_extracurricular = st.sidebar.selectbox('¿Participa en Extracurriculares?',
        [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')

    # Variables derivadas (flags automáticos)
    flag_cero_visitas = 1 if visitas_tutoria_semestre == 0 else 0
    flag_cero_asistencia = 1 if asistencia_porcentaje < 50 else 0

    # Construcción del DataFrame de entrada
    input_data = {
        'semestre_actual': semestre_actual,
        'promedio_notas': promedio_notas,
        'asistencia_porcentaje': asistencia_porcentaje,
        'materias_reprobadas_acum': materias_reprobadas_acum,
        'trabaja': trabaja,
        'horas_trabajo_semanal': horas_trabajo_semanal,
        'actividad_canvas_semanal': actividad_canvas_semanal,
        'tiene_beca': tiene_beca,
        'deuda_pendiente_usd': deuda_pendiente_usd,
        'vive_fuera_de_quito': vive_fuera_de_quito,
        'visitas_tutoria_semestre': visitas_tutoria_semestre,
        'participa_extracurricular': participa_extracurricular,
        'flag_cero_visitas_tutoria_semestre': flag_cero_visitas,
        'flag_cero_asistencia_porcentaje': flag_cero_asistencia,
    }

    # One-Hot Encoding manual para carrera (drop_first: Informatica es referencia)
    for c in ['Negocios_Digitales', 'Software', 'TI']:
        input_data[f'carrera_{c}'] = 1 if carrera == c else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feat, fill_value=0)

    # Predicción
    prob = modelo.predict_proba(input_df)[0][1]

    # Reglas de negocio
    riesgo_regla = ''
    if promedio_notas < 6.5 and tiene_beca == 0 and asistencia_porcentaje < 60:
        riesgo_regla = 'CRÍTICO'
    elif promedio_notas <= 7.0 or horas_trabajo_semanal > 20:
        riesgo_regla = 'MODERADO'

    # Clasificación por niveles de riesgo y texto coloreado
    if prob > 0.50 or riesgo_regla == 'CRÍTICO':
        estado_label = "Riesgo Crítico"
        tipo_alerta = "error"
        texto_prediccion = "El modelo predice que este estudiante :red[va a desertar]"
    elif prob >= 0.30 or riesgo_regla == 'MODERADO':
        estado_label = "Riesgo Moderado"
        tipo_alerta = "warning"
        texto_prediccion = "El modelo predice que este estudiante :orange[va a desertar]"
    else:
        estado_label = "Riesgo Bajo"
        tipo_alerta = "success"
        texto_prediccion = "El modelo predice que este estudiante :green[no va a desertar]"

    # Visualización de resultados
    st.header('Resultado de la Predicción')
    st.markdown(f"### {texto_prediccion}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Probabilidad de Deserción', f'{prob:.1%}')
    with col2:
        if tipo_alerta == "error":
            st.error(f'Estado: {estado_label}')
        elif tipo_alerta == "warning":
            st.warning(f'Estado: {estado_label}')
        else:
            st.success(f'Estado: {estado_label}')

    # Barra de progreso
    st.markdown('### Nivel de Riesgo')
    st.progress(min(prob, 1.0))

    # Resumen del estudiante
    st.markdown('---')
    st.subheader('Resumen del Estudiante')
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'''
        - **Carrera:** {carrera}
        - **Semestre:** {semestre_actual}
        - **Promedio:** {promedio_notas:.1f}
        - **Asistencia:** {asistencia_porcentaje:.1f}%
        - **Materias Reprobadas:** {materias_reprobadas_acum}
        ''')
    with col_b:
        st.markdown(f'''
        - **Beca:** {'Sí' if tiene_beca else 'No'}
        - **Deuda:** ${deuda_pendiente_usd:,.0f} USD
        - **Trabaja:** {'Sí' if trabaja else 'No'} ({horas_trabajo_semanal}h/sem)
        - **Canvas:** {actividad_canvas_semanal} interacciones/sem
        - **Tutorías:** {visitas_tutoria_semestre} visitas/sem
        ''')

    # Importancia de características
    st.markdown('---')
    st.subheader('Importancia de las Características del Modelo')
    importance_df = pd.DataFrame({
        'Feature': feat,
        'Importance': modelo.feature_importances_
    }).sort_values('Importance', ascending=True)
    st.bar_chart(importance_df.set_index('Feature'))

# --- Tab 2: Scoring batch ---
with tab2:
    st.subheader('Scoring en Batch - Cargar CSV de Estudiantes')
    archivo = st.file_uploader('Sube el CSV de estudiantes', type='csv')

    if archivo is not None:
        df_batch = pd.read_csv(archivo)
        X_batch = df_batch.drop(
            columns=[c for c in ['deserto_semestre'] if c in df_batch.columns],
            errors='ignore'
        )
        # One-Hot Encoding
        cat_cols = X_batch.select_dtypes(include=['object']).columns
        X_batch = pd.get_dummies(X_batch, columns=cat_cols, drop_first=True)
        X_batch = X_batch.reindex(columns=feat, fill_value=0)

        probs = modelo.predict_proba(X_batch)[:, 1]

        def clasificar_riesgo_batch(p):
            if p > 0.50:
                return 'Riesgo Crítico'
            elif p >= 0.30:
                return 'Riesgo Moderado'
            else:
                return 'Riesgo Bajo'

        df_batch['prob_desercion'] = probs.round(4)
        df_batch['nivel_riesgo'] = [clasificar_riesgo_batch(p) for p in probs]

        col1, col2, col3 = st.columns(3)
        col1.metric('Estudiantes totales', len(df_batch))
        col2.metric('En riesgo (>= 30%)', int((probs >= 0.30).sum()))
        col3.metric('% en riesgo', f'{(probs >= 0.30).mean():.1%}')

        st.dataframe(
            df_batch.sort_values('prob_desercion', ascending=False),
            use_container_width=True
        )
    else:
        st.info('Sube el archivo uisek_desercion_CLEAN.csv para ver '
                'las predicciones de toda la cartera.')

# --- Footer ---
st.markdown('---')
st.caption('Sistema de Alerta Temprana - Universidad Internacional SEK (UISEK) | '
           'Modelo: Random Forest | Umbral: 0.30 | '
           'Artefactos: rf_desercion_uisek.joblib | '
           'Desarrollado por: Grupo 5 - Data Science')
