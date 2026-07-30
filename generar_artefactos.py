import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('uisek_desercion_CLEAN.csv')

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

joblib.dump(rand_forest, 'rf_desercion_uisek.joblib')
joblib.dump(scaler, 'scaler_uisek.joblib')
joblib.dump(list(X.columns), 'feature_names_uisek.joblib')

print("Artefactos exportados exitosamente:")
print("   - rf_desercion_uisek.joblib")
print("   - scaler_uisek.joblib")
print("   - feature_names_uisek.joblib")
