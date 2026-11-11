import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLUNAS_CATEGORICAS = ['CHAS']
COLUNAS_QUANTITATIVAS = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
                         'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
COLUNAS = COLUNAS_QUANTITATIVAS + COLUNAS_CATEGORICAS
SEMENTE = 42


def encontrar_cotovelo(ks, inercias):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inercias) - inercias[-1]) / (inercias[0] - inercias[-1])
    distancias = np.abs(x + y - 1) / np.sqrt(2)
    return int(ks[int(np.argmax(distancias))])


df = pd.read_csv('HousingData.csv', na_values=['NA', ''])[COLUNAS]

imputer = ColumnTransformer([
    ('quantitativas', SimpleImputer(strategy='median'), COLUNAS_QUANTITATIVAS),
    ('categoricas', SimpleImputer(strategy='most_frequent'), COLUNAS_CATEGORICAS),
])
dados_imputados = imputer.fit_transform(df)

n_quant = len(COLUNAS_QUANTITATIVAS)
scaler = StandardScaler()
quantitativas_normalizadas = scaler.fit_transform(dados_imputados[:, :n_quant])
X = np.hstack([quantitativas_normalizadas, dados_imputados[:, n_quant:]])

ks = list(range(1, 11))
inercias = []
for k in ks:
    modelo = KMeans(n_clusters=k, random_state=SEMENTE, n_init=10)
    modelo.fit(X)
    inercias.append(modelo.inertia_)

print('Inércias por número de clusters (K):')
for k, inercia in zip(ks, inercias):
    print(f'K={k}: {inercia:.2f}')

k_otimo = encontrar_cotovelo(ks, inercias)
print(f'\nK ótimo pelo método do cotovelo: {k_otimo}')

plt.figure(figsize=(7, 4))
plt.plot(ks, inercias, 'o-')
plt.axvline(k_otimo, color='r', linestyle='--', label=f'K = {k_otimo}')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do cotovelo')
plt.legend()
plt.tight_layout()
plt.savefig('cotovelo.png', dpi=120)
plt.close()

kmeans_final = KMeans(n_clusters=k_otimo, random_state=SEMENTE, n_init=10)
kmeans_final.fit(X)

joblib.dump(kmeans_final, 'kmeans_housing.pkl')
joblib.dump(scaler, 'scaler_housing.pkl')
joblib.dump(imputer, 'imputer_housing.pkl')
joblib.dump({'quantitativas': COLUNAS_QUANTITATIVAS,
             'categoricas': COLUNAS_CATEGORICAS}, 'colunas_housing.pkl')
print('\nModelo, scaler, imputer e colunas salvos com sucesso!')
