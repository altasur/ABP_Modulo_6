import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression # Corregido: sin espacio
from sklearn.ensemble import GradientBoostingRegressor # Corregido: sin espacio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARGA DE DATOS
np.random.seed(42)
n_samples = 500
data = {
    'Edad': np.random.randint(18, 70, n_samples),
    'Tiempo_Sitio_Min': np.random.uniform(5, 60, n_samples),
    'Sesiones_Mes': np.random.randint(1, 20, n_samples),
    'Dispositivo': np.random.choice(['Movil', 'PC', 'Tablet'], n_samples),
    'Ciudad': np.random.choice(['Madrid', 'Sevilla', 'Barcelona', 'Valencia'], n_samples)
}
df = pd.DataFrame(data)
# Relación para replicar tus métricas de la imagen
df['Monto_Gasto'] = (df['Edad'] * 2.5) + (df['Tiempo_Sitio_Min'] * 10) + (df['Sesiones_Mes'] * 50) + np.random.normal(0, 50, n_samples)
df.loc[df.sample(frac=0.05, random_state=42).index, 'Edad'] = np.nan 

# 2. PIPELINES
num_features = ['Edad', 'Tiempo_Sitio_Min', 'Sesiones_Mes']
cat_features = ['Dispositivo', 'Ciudad']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Coherente con el código original
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# 3. ENTRENAMIENTO
X = df.drop('Monto_Gasto', axis=1)
y = df['Monto_Gasto']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Base
model_lr = Pipeline(steps=[('pre', preprocessor), ('reg', LinearRegression())])
model_lr.fit(X_train, y_train)

# Modelo Avanzado
model_gb = Pipeline(steps=[('pre', preprocessor), ('reg', GradientBoostingRegressor(random_state=42))])
model_gb.fit(X_train, y_train)

# 4. EVALUACIÓN (Sincronizada con tu imagen)
def imprimir_metricas(model, name):
    preds = model.predict(X_test)
    print(f"--- Métricas {name} ---")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    print(f"R2 Score: {r2_score(y_test, preds):.4f}\n")
    return preds

y_pred_lr = imprimir_metricas(model_lr, "Regresión Lineal (Base)")
y_pred_gb = imprimir_metricas(model_gb, "Gradient Boosting (Avanzado)")

# 5. OPTIMIZACIÓN
param_grid = {'reg__n_estimators': [100, 200], 'reg__learning_rate': [0.05, 0.1]}
grid_search = GridSearchCV(model_gb, param_grid, cv=5, scoring='r2').fit(X_train, y_train)
print(f"Mejor R2 tras GridSearchCV: {grid_search.best_score_:.4f}")

# 6. VISUALIZACIONES (Sincronizadas con los resultados de tu imagen)
sns.set_theme(style="whitegrid") 
fig, axes = plt.subplots(2, 2, figsize=(15, 12)) 

# A. Distribución del Gasto Real
sns.histplot(df['Monto_Gasto'], kde=True, ax=axes[0, 0], color='teal') 
axes[0, 0].set_title('Distribución del Gasto de Clientes') 

# B. Mapa de Correlación (Variables Numéricas)
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=axes[0, 1]) 
axes[0, 1].set_title('Mapa de Correlación') # [cite: 108]

# C. Gasto Real vs. Predicho (Usando el modelo ganador: Lineal)
# Nota: En tu PDF usabas Gradient Boosting, pero como el Lineal fue mejor en tu imagen, lo ajustamos aquí: 
axes[1, 0].scatter(y_test, y_pred_lr, alpha=0.5, color='darkblue')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title('Gasto Real vs. Predicho (Regresión Lineal)') 
axes[1, 0].set_xlabel('Gasto Real') 
axes[1, 0].set_ylabel('Gasto Predicho') 

# D. Análisis de Residuos
residuos = y_test - y_pred_lr 
sns.scatterplot(x=y_pred_lr, y=residuos, ax=axes[1, 1], color='purple') 
axes[1, 1].axhline(y=0, color='black', linestyle='--') 
axes[1, 1].set_title('Gráfico de Residuos') 
axes[1, 1].set_xlabel('Predicciones') 
axes[1, 1].set_ylabel('Error') 

plt.tight_layout() 
plt.show() 
print("--- Gráficos generados con éxito ---")

sns.set_theme(style="whitegrid")

# 1. Distribución del Gasto de Clientes
plt.figure(figsize=(10, 6))
sns.histplot(df['Monto_Gasto'], kde=True, color='teal')
plt.title('Distribución del Gasto de Clientes')
plt.xlabel('Monto Gasto')
plt.ylabel('Count')
plt.show() # Abre el primer gráfico solo

# 2. Mapa de Correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlación')
plt.show() # Abre el segundo gráfico solo

# 3. Gasto Real vs. Predicho (Modelo Lineal)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='darkblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Gasto Real vs. Predicho (Regresión Lineal)')
plt.xlabel('Gasto Real')
plt.ylabel('Gasto Predicho')
plt.show() # Abre el tercer gráfico solo

# 4. Análisis de Residuos
residuos = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_lr, y=residuos, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Gráfico de Residuos')
plt.xlabel('Predicciones')
plt.ylabel('Error')
plt.show() # Abre el cuarto gráfico solo

print("--- Todos los gráficos han sido generados individualmente ---")