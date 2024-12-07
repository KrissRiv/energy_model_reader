import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configuración de estilo para las visualizaciones
sns.set(style="whitegrid")

# Directorio de datos
DATASET_DIRECTORY = "./data/"

def read_dataset(dataset_name):
    # Función para leer el conjunto de datos a partir de un archivo CSV
    return pd.read_csv(DATASET_DIRECTORY + dataset_name)

# Cargar el conjunto de datos
df_energy = read_dataset("owid-energy-data.csv")

# Comprensión del modelo: nombre de columnas y tipo de datos
print("\nInformación del DataFrame:")
df_energy.info(verbose=True)
print("\nColumnas del DataFrame:")
print(df_energy.columns)
print("\nPrimeras filas del DataFrame:")
print(df_energy.head())

# Evaluación de datos nulos
print("\nConteo de valores nulos por columna:")
print(df_energy.isna().sum())

# Eliminación de filas con muchos valores nulos
df = df_energy.dropna(axis=0, thresh=6)
print("\nConteo de valores nulos tras eliminación:")
print(df.isna().sum())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df_energy.describe())

# Evaluación de la correlación
cols_mean = [col for col in df_energy if 'mean' in col]
cols_error = [col for col in df_energy if 'error' in col]
cols_worst = [col for col in df_energy if 'worst' in col]

selected_cols = cols_mean + cols_error + cols_worst
missing_cols = [col for col in selected_cols if col not in df_energy.columns]

if missing_cols:
    print(f"\nAdvertencia: Las siguientes columnas no están en el DataFrame: {missing_cols}")
    selected_cols = [col for col in selected_cols if col in df_energy.columns]

if not df_energy[selected_cols].empty:
    corr_matrix = df_energy[selected_cols].corr()
    if not corr_matrix.isnull().values.all():
        plt.figure(figsize=(16, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='viridis')
        plt.title('Matriz de Correlación')
        plt.show()

# Agrupamiento de la información por países
print("\nTamaño del grupo por país:")
print(df_energy.groupby("country").size())

# Relleno de valores nulos con cero
df_energy = df_energy.fillna(0)
print("\nConteo de valores nulos tras relleno:")
print(df_energy.isnull().sum())

# Filtro de datos desde el año 2000
partial_df = (df_energy["year"] > 2000) & (df_energy["country"] != "World")
filtered_df = df_energy[partial_df]

# Histograma de los datos filtrados
sns.histplot(data=filtered_df, x='year')
plt.title('Distribución de Años en el DataFrame Filtrado')
plt.show()

# Ordenamiento basado en el PIB
df_gdp = df_energy[(df_energy['gdp']>0) & (df_energy['country']!='World') & (df_energy['year'].between(2000, 2023))]
country_gdp = df_gdp.sort_values(by='gdp', ascending=True)

# Segmentación de países de Sudamérica
south_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
df_south_america = df_gdp[df_gdp['country'].isin(south_america)]

# Gráfico de PIB de países sudamericanos
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df_south_america[df_south_america['gdp'].notna()], x='country', y='gdp', ax=ax)
ax.set_title('PIB por País en Sudamérica')
ax.set_xlabel('País')
ax.set_ylabel('PIB')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Consumo energético por tipo en Sudamérica
df_south_america[['biofuel_consumption', 'coal_consumption', 'gas_consumption', 'hydro_consumption', 'renewables_consumption', 'solar_consumption', 'wind_consumption']].hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograma del Consumo Energético por Tipo')
plt.tight_layout()
plt.show()

# Gráfico de pastel de consumo energético
fig, ax = plt.subplots(figsize=(10, 6))
source_cols = ['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption', 'nuclear_consumption']
df_south_america[df_south_america['gdp'].notna()][source_cols].sum().plot(kind='pie', ax=ax, autopct='%1.1f%%')
ax.set_title('Consumo Energético por Fuente')
ax.set_ylabel('')
plt.tight_layout()
plt.show()

# Análisis del consumo energético basado en el PIB
plt.figure(figsize=(15, 7))
for country in df_south_america['country'].unique():
    country_data = df_south_america[df_south_america['country'] == country]
    plt.plot(country_data['year'], country_data['energy_per_gdp'], label=country)
plt.title('Consumo Energético por PIB: 2000-2020')
plt.xlabel('Año')
plt.ylabel('Consumo Energético por PIB')
plt.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Evolución del PIB en Sudamérica
plt.figure(figsize=(15, 7))
for country in df_south_america['country'].unique():
    country_data = df_south_america[df_south_america['country'] == country]
    plt.plot(country_data['year'], country_data['gdp'], label=country)
plt.title('Evolución del PIB en Sudamérica: 2000-2020')
plt.xlabel('Año')
plt.ylabel('PIB')
plt.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Seleccionar los 5 países con mayor PIB en 2020
top_5_countries = df_energy[df_energy['year'] == 2020].nlargest(5, 'gdp')['country']
top_5_countries_df = df_energy[df_energy['country'].isin(top_5_countries)]

# Preparación de los datos para el modelo
X = top_5_countries_df[['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption', 'nuclear_consumption']]
y = top_5_countries_df['gdp']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento de modelos
tree_reg = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_reg.fit(X_train, y_train)

rf_est = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf_est.fit(X_train, y_train)

gb_est = GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42)
gb_est.fit(X_train, y_train)

# Evaluación de modelos
models = [tree_reg, rf_est, gb_est]

for model in models:
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print(f"Model: {model.__class__.__name__}")
    print(f"MAE Training: {mean_absolute_error(y_train, y_train_pred)}")
    print(f"R2 Training: {r2_score(y_train, y_train_pred)}")
    print(f"MAE Test: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 Test: {r2_score(y_test, y_pred)}")
    print()

# Búsqueda de hiperparámetros para RandomForest
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_model_rf = grid_search_rf.best_estimator_

y_pred = best_model_rf.predict(X_test)
y_train_pred = best_model_rf.predict(X_train)
print(f"Model: {best_model_rf.__class__.__name__}")
print(f"MAE Training: {mean_absolute_error(y_train, y_train_pred)}")
print(f"R2 Training: {r2_score(y_train, y_train_pred)}")
print(f"MAE Test: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Test: {r2_score(y_test, y_pred)}")
print()

# Búsqueda de hiperparámetros para GradientBoosting
param_grid_gr = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1]
}

grid_search_gr = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gr, cv=5, n_jobs=-1)
grid_search_gr.fit(X_train, y_train)

best_model_gr = grid_search_gr.best_estimator_

y_pred = best_model_gr.predict(X_test)
y_train_pred = best_model_gr.predict(X_train)
print(f"Model: {best_model_gr.__class__.__name__}")
print(f"MAE Training: {mean_absolute_error(y_train, y_train_pred)}")
print(f"R2 Training: {r2_score(y_train, y_train_pred)}")
print(f"MAE Test: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Test: {r2_score(y_test, y_pred)}")
print()

# Guardar el mejor modelo en un archivo
joblib.dump(best_model_rf, 'sa_energy_model.pkl')
