import joblib
import numpy as np
import pandas as pd

kmeans = joblib.load('kmeans_housing.pkl')
scaler = joblib.load('scaler_housing.pkl')
imputer = joblib.load('imputer_housing.pkl')
colunas = joblib.load('colunas_housing.pkl')

quantitativas = colunas['quantitativas']
categoricas = colunas['categoricas']

nova_residencia = {
    'CRIM': 0.05,
    'ZN': 18.0,
    'INDUS': 2.31,
    'CHAS': 0.0,
    'NOX': 0.538,
    'RM': 6.5,
    'AGE': 65.2,
    'DIS': 4.09,
    'RAD': 1,
    'TAX': 296,
    'PTRATIO': 15.3,
    'B': 396.90,
    'LSTAT': 4.98,
    'MEDV': 25.0
}

nova_df = pd.DataFrame([nova_residencia])[quantitativas + categoricas]

nova_imputada = imputer.transform(nova_df)
n_quant = len(quantitativas)
nova_normalizada = np.hstack([scaler.transform(nova_imputada[:, :n_quant]),
                              nova_imputada[:, n_quant:]])

cluster_inferido = int(kmeans.predict(nova_normalizada)[0])
print(f'A nova residência pertence ao Grupo (Cluster): {cluster_inferido}')

centroide = kmeans.cluster_centers_[cluster_inferido]
centroide_quantitativas = scaler.inverse_transform(centroide[None, :n_quant])[0]
centroide_desnormalizado = pd.Series(
    np.concatenate([centroide_quantitativas, centroide[n_quant:]]),
    index=quantitativas + categoricas,
)

print('\n--- Dados Desnormalizados do Centróide Inferido ---')
for coluna, valor in centroide_desnormalizado.items():
    print(f'{coluna}: {valor:.4f}')
