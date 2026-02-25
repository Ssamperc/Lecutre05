import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuración de la página
st.set_page_config(page_title="Iris Classifier Explorer", layout="wide")

st.title("🌿 Iris Dataset: Explorador de Modelos y Fronteras")
st.markdown("""
Esta app permite explorar diferentes clasificadores, ajustar sus hiperparámetros 
y visualizar cómo separan las especies de flores.
""")

# --- SIDEBAR: CONFIGURACIÓN ---
st.sidebar.header("Configuración del Modelo")

dataset_name = st.sidebar.selectbox("Selecciona el Dataset", ("Iris",))
classifier_name = st.sidebar.selectbox(
    "Selecciona el Clasificador", 
    ("KNN", "SVM", "Random Forest")
)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K (n_neighbors)", 1, 15, 5)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 2, 15, 5)
        n_estimators = st.sidebar.slider("N Estimators", 1, 100, 10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.feature_names, iris.target_names

X, y, feature_names, target_names = load_data()

# --- LÓGICA DEL MODELO ---
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], probability=True)
    else:
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"], 
            max_depth=params["max_depth"], 
            random_index=1234
        )
    return clf

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = get_classifier(classifier_name, params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- LAYOUT PRINCIPAL: MÉTRICAS ---
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Clasificador: {classifier_name}")
    st.write(f"**Precisión (Accuracy):** {acc:.2f}")
    
    # Matriz de Confusión
    st.write("**Matriz de Confusión:**")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    st.pyplot(fig_cm)

with col2:
    st.write("**Reporte de Clasificación:**")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

# --- FRONTERAS DE DECISIÓN ---
st.divider()
st.subheader("Visualización de Fronteras de Decisión")
st.info("Para visualizar las fronteras en 2D, aplicamos PCA (Análisis de Componentes Principales) sobre los datos.")

# Preparar datos para fronteras (PCA a 2D)
pca = PCA(2)
X_pca = pca.fit_transform(X)
clf_pca = get_classifier(classifier_name, params) # Re-entrenar con 2D
clf_pca.fit(X_pca, y)

# Crear malla para el gráfico
h = .02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig_dec, ax_dec = plt.subplots(figsize=(8, 6))
ax_dec.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax_dec.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='viridis')
ax_dec.set_xlabel('Componente Principal 1')
ax_dec.set_ylabel('Componente Principal 2')
legend1 = ax_dec.legend(*scatter.legend_elements(), title="Clases")
ax_dec.add_artist(legend1)

st.pyplot(fig_dec)
