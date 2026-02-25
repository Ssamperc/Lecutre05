import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #3ECFCF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #aaa; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #fff; font-size: 1.8rem; font-weight: 700; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        border-left: 4px solid #6C63FF;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.8rem;
        color: #ddd;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────
@st.cache_data
def get_iris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = [iris.target_names[t] for t in iris.target]
    return iris, df

iris, df_iris = get_iris()
FEATURE_NAMES = iris.feature_names
CLASS_NAMES   = list(iris.target_names)
COLORS        = ["#6C63FF", "#3ECFCF", "#FF6B6B"]

# ─────────────────────────────────────────────
# MODELOS DISPONIBLES
# ─────────────────────────────────────────────
MODELS = {
    "Logistic Regression":       LogisticRegression(max_iter=1000),
    "Decision Tree":             DecisionTreeClassifier(),
    "Random Forest":             RandomForestClassifier(n_estimators=100),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=100),
    "SVM (RBF)":                 SVC(probability=True),
    "SVM (Linear)":              SVC(kernel="linear", probability=True),
    "K-Nearest Neighbors":       KNeighborsClassifier(),
    "Naive Bayes":               GaussianNB(),
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 🤖 Modelos")
    selected_models = st.multiselect(
        "Selecciona modelos",
        list(MODELS.keys()),
        default=["Logistic Regression", "SVM (RBF)", "Random Forest"]
    )

    st.markdown("### 📊 Features para fronteras")
    feat1 = st.selectbox("Feature X", FEATURE_NAMES, index=0)
    feat2 = st.selectbox("Feature Y", FEATURE_NAMES, index=1)

    st.markdown("### 🔀 División del dataset")
    test_size  = st.slider("Tamaño del test (%)", 10, 40, 20) / 100
    random_seed = st.number_input("Semilla aleatoria", 0, 999, 42)

    st.markdown("### 📐 Métricas")
    avg_method = st.selectbox("Promedio para métricas", ["macro", "weighted", "micro"], index=0)

    st.markdown("### 🔁 Validación cruzada")
    do_cv   = st.checkbox("Activar CV", value=True)
    cv_folds = st.slider("Folds", 2, 10, 5) if do_cv else 5

    st.markdown("### 🗺️ Opciones de visualización")
    mesh_resolution = st.slider("Resolución de frontera", 50, 300, 150)
    show_pca_plot   = st.checkbox("Proyección PCA 2D", value=True)
    show_feature_imp = st.checkbox("Importancia de features", value=True)

    run_btn = st.button("🚀 Entrenar y Evaluar", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# TÍTULO
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌸 Iris Classifier Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clasificación multimodelo con análisis de desempeño y fronteras de decisión</div>', unsafe_allow_html=True)

if not selected_models:
    st.warning("⚠️ Selecciona al menos un modelo en la barra lateral.")
    st.stop()

# ─────────────────────────────────────────────
# PREPARAR DATOS
# ─────────────────────────────────────────────
X = iris.data
y = iris.target

fi1 = FEATURE_NAMES.index(feat1)
fi2 = FEATURE_NAMES.index(feat2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# ENTRENAR MODELOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_all(selected, test_size, random_seed, avg_method, do_cv, cv_folds):
    results = {}
    X_full = iris.data
    y_full = iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_seed, stratify=y_full
    )
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc  = sc.transform(X_te)

    for name in selected:
        model = MODELS[name]
        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)

        metrics = {
            "Accuracy":  accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, average=avg_method, zero_division=0),
            "Recall":    recall_score(y_te, y_pred, average=avg_method, zero_division=0),
            "F1-Score":  f1_score(y_te, y_pred, average=avg_method, zero_division=0),
            "MCC":       matthews_corrcoef(y_te, y_pred),
            "Kappa":     cohen_kappa_score(y_te, y_pred),
        }
        cv_scores = None
        if do_cv:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", MODELS[name])])
            cv_scores = cross_val_score(pipe, X_full, y_full, cv=StratifiedKFold(cv_folds), scoring="accuracy")

        results[name] = {
            "model":     model,
            "scaler":    sc,
            "metrics":   metrics,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "cv_scores": cv_scores,
            "X_te_sc":   X_te_sc,
            "y_te":      y_te,
        }
    return results, X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, sc

with st.spinner("Entrenando modelos..."):
    results, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = train_all(
        tuple(selected_models), test_size, random_seed, avg_method, do_cv, cv_folds
    )

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Métricas de Desempeño",
    "🗺️ Fronteras de Decisión",
    "📈 Curvas ROC & PR",
    "🔥 Matrices de Confusión",
    "🔬 Análisis Avanzado",
])

# ═══════════════════════════════════════════════
# TAB 1 – MÉTRICAS
# ═══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Comparativa de Métricas</div>', unsafe_allow_html=True)

    # Tabla resumen
    rows = []
    for name, r in results.items():
        row = {"Modelo": name}
        row.update({k: f"{v:.4f}" for k, v in r["metrics"].items()})
        if r["cv_scores"] is not None:
            row["CV Mean±Std"] = f"{r['cv_scores'].mean():.4f} ± {r['cv_scores'].std():.4f}"
        rows.append(row)

    df_metrics = pd.DataFrame(rows).set_index("Modelo")
    st.dataframe(
        df_metrics.style
            .background_gradient(cmap="RdYlGn", subset=["Accuracy","Precision","Recall","F1-Score"])
            .format(precision=4),
        use_container_width=True
    )

    # Gráfica comparativa
    st.markdown('<div class="section-header">Gráfica Comparativa por Métrica</div>', unsafe_allow_html=True)
    metric_sel = st.multiselect(
        "Métricas a graficar",
        ["Accuracy","Precision","Recall","F1-Score","MCC","Kappa"],
        default=["Accuracy","F1-Score","Precision","Recall"]
    )

    if metric_sel:
        plot_data = {m: [] for m in metric_sel}
        model_names = list(results.keys())
        for name, r in results.items():
            for m in metric_sel:
                plot_data[m].append(r["metrics"][m])

        fig = go.Figure()
        for m in metric_sel:
            fig.add_trace(go.Bar(name=m, x=model_names, y=plot_data[m], text=[f"{v:.3f}" for v in plot_data[m]], textposition="outside"))
        fig.update_layout(
            barmode="group", height=420,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#ddd",
            yaxis=dict(range=[0, 1.12], gridcolor="#333"),
            xaxis=dict(gridcolor="#333"),
            legend=dict(bgcolor="#1e1e2e"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # CV box plots
    if do_cv:
        st.markdown('<div class="section-header">Distribución Cross-Validation (Accuracy)</div>', unsafe_allow_html=True)
        fig_cv = go.Figure()
        for name, r in results.items():
            if r["cv_scores"] is not None:
                fig_cv.add_trace(go.Box(y=r["cv_scores"], name=name, boxpoints="all", jitter=0.4))
        fig_cv.update_layout(
            height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#ddd", yaxis=dict(title="Accuracy", gridcolor="#333"),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2 – FRONTERAS DE DECISIÓN
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Fronteras de Decisión (2 Features)</div>', unsafe_allow_html=True)
    st.info(f"Usando features: **{feat1}** (X) vs **{feat2}** (Y)")

    X2 = X[:, [fi1, fi2]]
    X2_train, X2_test, _, _ = train_test_split(X2, y, test_size=test_size, random_state=random_seed, stratify=y)
    sc2 = StandardScaler()
    X2_train_sc = sc2.fit_transform(X2_train)
    X2_test_sc  = sc2.transform(X2_test)

    # Mesh grid
    x_min, x_max = X2_train_sc[:,0].min()-0.5, X2_train_sc[:,0].max()+0.5
    y_min, y_max = X2_train_sc[:,1].min()-0.5, X2_train_sc[:,1].max()+0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, mesh_resolution),
        np.linspace(y_min, y_max, mesh_resolution)
    )

    n_models = len(selected_models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig_db, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig_db.patch.set_facecolor("#0e1117")
    axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

    cmap_bg = plt.cm.RdYlBu
    cmap_pts = [COLORS[c] for c in sorted(set(y))]

    for idx, name in enumerate(selected_models):
        ax = axes_flat[idx]
        ax.set_facecolor("#1a1a2e")

        m2 = MODELS[name].__class__(**MODELS[name].get_params())
        m2.fit(X2_train_sc, X2_train)
        Z = m2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=[-0.5,0.5,1.5,2.5])
        ax.contour(xx, yy, Z, colors="white", linewidths=0.8, alpha=0.5, levels=[0.5,1.5])

        for cls, col in zip(range(3), COLORS):
            mask_tr = y_train == cls
            mask_te = y_test  == cls
            Xtr_cls = sc2.transform(X2_train)[mask_tr]
            Xte_cls = sc2.transform(X2_test)[mask_te]
            ax.scatter(Xtr_cls[:,0], Xtr_cls[:,1], c=col, s=40, edgecolors="white", linewidths=0.5, alpha=0.85)
            ax.scatter(Xte_cls[:,0],  Xte_cls[:,1],  c=col, s=80, marker="*", edgecolors="black", linewidths=0.6)

        patches = [mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i]) for i in range(3)]
        ax.legend(handles=patches, fontsize=7, loc="upper left",
                  facecolor="#1e1e2e", edgecolor="#444", labelcolor="white")

        acc = results[name]["metrics"]["Accuracy"]
        ax.set_title(f"{name}\nAcc={acc:.3f}", color="white", fontsize=10, pad=6)
        ax.set_xlabel(feat1, color="#aaa", fontsize=8)
        ax.set_ylabel(feat2, color="#aaa", fontsize=8)
        ax.tick_params(colors="#777")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_db)

    # PCA 2D projection
    if show_pca_plot:
        st.markdown('<div class="section-header">Proyección PCA 2D</div>', unsafe_allow_html=True)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_sc)
        X_test_pca = pca.transform(X_test_sc)

        fig_pca = go.Figure()
        for cls_i, col in zip(range(3), COLORS):
            mask = y_train == cls_i
            fig_pca.add_trace(go.Scatter(
                x=X_pca[mask,0], y=X_pca[mask,1],
                mode="markers", name=f"{CLASS_NAMES[cls_i]} (train)",
                marker=dict(color=col, size=7, opacity=0.7)
            ))
            mask_te = y_test == cls_i
            fig_pca.add_trace(go.Scatter(
                x=X_test_pca[mask_te,0], y=X_test_pca[mask_te,1],
                mode="markers", name=f"{CLASS_NAMES[cls_i]} (test)",
                marker=dict(color=col, size=11, symbol="star", opacity=1)
            ))
        var_exp = pca.explained_variance_ratio_
        fig_pca.update_layout(
            height=450,
            xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#ddd",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3 – ROC & PR CURVES
# ═══════════════════════════════════════════════
with tab3:
    # ROC
    st.markdown('<div class="section-header">Curvas ROC – One-vs-Rest</div>', unsafe_allow_html=True)
    y_bin = label_binarize(y_test, classes=[0,1,2])

    cols_roc = st.columns(len(selected_models))
    for col_idx, name in enumerate(selected_models):
        with cols_roc[col_idx]:
            r = results[name]
            fig_roc = go.Figure()
            for cls_i, cls_name in enumerate(CLASS_NAMES):
                fpr, tpr, _ = roc_curve(y_bin[:,cls_i], r["y_prob"][:,cls_i])
                roc_auc_val = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{cls_name} (AUC={roc_auc_val:.2f})",
                    line=dict(color=COLORS[cls_i], width=2)
                ))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                line=dict(dash="dash", color="#555"), showlegend=False))
            fig_roc.update_layout(
                title=name, height=350,
                xaxis_title="FPR", yaxis_title="TPR",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#ddd",
                legend=dict(font_size=9, bgcolor="#1e1e2e"),
                margin=dict(t=40, b=40, l=40, r=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # PR Curves
    st.markdown('<div class="section-header">Curvas Precision-Recall</div>', unsafe_allow_html=True)
    cols_pr = st.columns(len(selected_models))
    for col_idx, name in enumerate(selected_models):
        with cols_pr[col_idx]:
            r = results[name]
            fig_pr = go.Figure()
            for cls_i, cls_name in enumerate(CLASS_NAMES):
                prec, rec, _ = precision_recall_curve(y_bin[:,cls_i], r["y_prob"][:,cls_i])
                ap = average_precision_score(y_bin[:,cls_i], r["y_prob"][:,cls_i])
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines", name=f"{cls_name} (AP={ap:.2f})",
                    line=dict(color=COLORS[cls_i], width=2)
                ))
            fig_pr.update_layout(
                title=name, height=350,
                xaxis_title="Recall", yaxis_title="Precision",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#ddd",
                legend=dict(font_size=9, bgcolor="#1e1e2e"),
                margin=dict(t=40, b=40, l=40, r=10),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4 – MATRICES DE CONFUSIÓN
# ═══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Matrices de Confusión</div>', unsafe_allow_html=True)

    ncols_cm = min(3, len(selected_models))
    nrows_cm = (len(selected_models) + ncols_cm - 1) // ncols_cm
    fig_cm, axes_cm = plt.subplots(nrows_cm, ncols_cm, figsize=(5*ncols_cm, 4*nrows_cm))
    fig_cm.patch.set_facecolor("#0e1117")
    axes_cm_flat = np.array(axes_cm).flatten() if len(selected_models) > 1 else [axes_cm]

    for idx, name in enumerate(selected_models):
        ax = axes_cm_flat[idx]
        ax.set_facecolor("#1a1a2e")
        cm = confusion_matrix(results[name]["y_te"], results[name]["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=cm, fmt="d", ax=ax,
            cmap="Blues", linewidths=0.5, linecolor="#333",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title(name, color="white", fontsize=10, pad=8)
        ax.set_xlabel("Predicho", color="#aaa", fontsize=8)
        ax.set_ylabel("Real", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=8)

    for idx in range(len(selected_models), len(axes_cm_flat)):
        axes_cm_flat[idx].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_cm)

    # Reports de texto
    with st.expander("📋 Ver reportes de clasificación completos"):
        for name, r in results.items():
            st.markdown(f"**{name}**")
            st.code(classification_report(r["y_te"], r["y_pred"], target_names=CLASS_NAMES))

# ═══════════════════════════════════════════════
# TAB 5 – ANÁLISIS AVANZADO
# ═══════════════════════════════════════════════
with tab5:

    # Feature Importance
    if show_feature_imp:
        st.markdown('<div class="section-header">Importancia / Coeficientes de Features</div>', unsafe_allow_html=True)
        imp_models = {n: r for n, r in results.items()
                      if hasattr(r["model"], "feature_importances_") or hasattr(r["model"], "coef_")}

        if imp_models:
            for name, r in imp_models.items():
                m = r["model"]
                if hasattr(m, "feature_importances_"):
                    imp = m.feature_importances_
                    title = f"{name} – Feature Importances"
                elif hasattr(m, "coef_"):
                    imp = np.mean(np.abs(m.coef_), axis=0)
                    title = f"{name} – |Coeficientes| promedio"

                fig_imp = go.Figure(go.Bar(
                    x=FEATURE_NAMES, y=imp,
                    marker_color=COLORS[0],
                    text=[f"{v:.4f}" for v in imp], textposition="outside"
                ))
                fig_imp.update_layout(
                    title=title, height=300,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#ddd",
                    yaxis=dict(gridcolor="#333"), margin=dict(t=40, b=40)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Ningún modelo seleccionado expone importancias o coeficientes.")

    # Scatter Matrix (Pair Plot) interactivo
    st.markdown('<div class="section-header">Scatter Matrix del Dataset</div>', unsafe_allow_html=True)
    fig_sp = px.scatter_matrix(
        df_iris,
        dimensions=FEATURE_NAMES,
        color="species",
        color_discrete_sequence=COLORS,
        opacity=0.75,
        height=600,
    )
    fig_sp.update_traces(diagonal_visible=False, showupperhalf=False)
    fig_sp.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#ddd",
    )
    st.plotly_chart(fig_sp, use_container_width=True)

    # Violin plots por feature
    st.markdown('<div class="section-header">Distribución de Features por Clase</div>', unsafe_allow_html=True)
    feat_viol = st.selectbox("Feature", FEATURE_NAMES, key="violin_feat")
    fig_viol = go.Figure()
    for cls_i, cls_name in enumerate(CLASS_NAMES):
        vals = df_iris[df_iris.target == cls_i][feat_viol]
        fig_viol.add_trace(go.Violin(
            y=vals, name=cls_name, box_visible=True, meanline_visible=True,
            line_color=COLORS[cls_i], fillcolor=COLORS[cls_i], opacity=0.55
        ))
    fig_viol.update_layout(
        height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#ddd",
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_viol, use_container_width=True)

    # Correlación de features
    st.markdown('<div class="section-header">Mapa de Correlación</div>', unsafe_allow_html=True)
    corr = df_iris[FEATURE_NAMES].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, height=380
    )
    fig_corr.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#ddd",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;'>"
    "Iris Classifier Studio · Powered by Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
