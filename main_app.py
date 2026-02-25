import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# DARK THEME HELPERS
# ──────────────────────────────────────────────────────────────
BG      = "#0e1117"
SURFACE = "#1a1a2e"
ACCENT  = "#6C63FF"
COLORS  = ["#6C63FF", "#3ECFCF", "#FF6B6B"]
TEXT    = "#dddddd"
GRID    = "#2a2a3e"

def dark_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig

def dark_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5)
    return ax

plt.rcParams.update({
    "text.color":        TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    GRID,
    "grid.color":        GRID,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  GRID,
    "legend.labelcolor": TEXT,
})

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title{
    font-size:2.3rem;font-weight:800;
    background:linear-gradient(135deg,#6C63FF,#3ECFCF);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.subtitle{color:#888;font-size:1rem;margin-bottom:1.2rem;}
.sec{
    font-size:1rem;font-weight:600;
    border-left:4px solid #6C63FF;
    padding-left:.6rem;margin:1.2rem 0 .6rem;color:#ddd;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATOS
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df   = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"]  = iris.target
    df["species"] = [iris.target_names[t] for t in iris.target]
    return iris, df

iris, df_iris   = load_data()
FEAT_NAMES      = list(iris.feature_names)
CLASS_NAMES     = list(iris.target_names)

# ──────────────────────────────────────────────────────────────
# MODELOS
# ──────────────────────────────────────────────────────────────
ALL_MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000),
    "Decision Tree":        DecisionTreeClassifier(),
    "Random Forest":        RandomForestClassifier(n_estimators=100),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100),
    "SVM (RBF)":            SVC(probability=True),
    "SVM (Linear)":         SVC(kernel="linear", probability=True),
    "K-Nearest Neighbors":  KNeighborsClassifier(),
    "Naive Bayes":          GaussianNB(),
}

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 🤖 Modelos")
    sel_models = st.multiselect(
        "Selecciona modelos",
        list(ALL_MODELS.keys()),
        default=["Logistic Regression", "SVM (RBF)", "Random Forest", "Decision Tree"],
    )

    st.markdown("### 📊 Features – Frontera de decisión")
    feat_x = st.selectbox("Feature X", FEAT_NAMES, index=0)
    feat_y = st.selectbox("Feature Y", FEAT_NAMES, index=1)

    st.markdown("### 🔀 Dataset split")
    test_pct    = st.slider("Tamaño del test (%)", 10, 40, 20)
    seed        = st.number_input("Semilla", 0, 999, 42)

    st.markdown("### 📐 Métricas")
    avg_method  = st.selectbox("Promedio", ["macro", "weighted", "micro"], index=0)

    st.markdown("### 🔁 Validación cruzada")
    do_cv    = st.checkbox("Activar CV", value=True)
    cv_folds = st.slider("Folds", 2, 10, 5) if do_cv else 5

    st.markdown("### 🗺️ Visualización")
    mesh_res    = st.slider("Resolución frontera", 50, 250, 120)
    show_pca    = st.checkbox("Proyección PCA 2D", value=True)
    show_imp    = st.checkbox("Importancia de features", value=True)

if not sel_models:
    st.warning("⚠️ Selecciona al menos un modelo.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# ENCABEZADO
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌸 Iris Classifier Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clasificación multimodelo · Desempeño · Fronteras de decisión</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO (cacheado)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelos…")
def run_training(sel, test_pct, seed, avg_method, do_cv, cv_folds):
    X, y     = iris.data, iris.target
    test_sz  = test_pct / 100
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_sz, random_state=seed, stratify=y)

    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)

    res = {}
    for name in sel:
        mdl = ALL_MODELS[name].__class__(**ALL_MODELS[name].get_params())
        mdl.fit(Xtr_sc, y_tr)
        yp   = mdl.predict(Xte_sc)
        yprob= mdl.predict_proba(Xte_sc)

        metrics = {
            "Accuracy":  accuracy_score(y_te, yp),
            "Precision": precision_score(y_te, yp, average=avg_method, zero_division=0),
            "Recall":    recall_score(y_te, yp, average=avg_method, zero_division=0),
            "F1-Score":  f1_score(y_te, yp, average=avg_method, zero_division=0),
            "MCC":       matthews_corrcoef(y_te, yp),
            "Kappa":     cohen_kappa_score(y_te, yp),
        }
        cv_sc = None
        if do_cv:
            pipe  = Pipeline([("sc", StandardScaler()),
                              ("clf", ALL_MODELS[name].__class__(**ALL_MODELS[name].get_params()))])
            cv_sc = cross_val_score(pipe, X, y,
                                    cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
                                    scoring="accuracy")

        res[name] = dict(model=mdl, scaler=sc,
                         metrics=metrics, yp=yp, yprob=yprob,
                         cv=cv_sc, Xte=Xte_sc, yte=y_te)

    return res, X_tr, X_te, y_tr, y_te, Xtr_sc, Xte_sc, sc

results, X_tr, X_te, y_tr, y_te, Xtr_sc, Xte_sc, sc = run_training(
    tuple(sel_models), test_pct, seed, avg_method, do_cv, cv_folds
)

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Métricas",
    "🗺️ Fronteras de Decisión",
    "📈 Curvas ROC & PR",
    "🔥 Matrices de Confusión",
    "🔬 Análisis Avanzado",
])

# ════════════════════════════════════════════════════════════
# TAB 1 ── MÉTRICAS
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">Tabla comparativa de métricas</div>', unsafe_allow_html=True)

    rows = []
    for name, r in results.items():
        row = {"Modelo": name}
        row.update({k: round(v, 4) for k, v in r["metrics"].items()})
        if r["cv"] is not None:
            row["CV Mean"] = round(r["cv"].mean(), 4)
            row["CV Std"]  = round(r["cv"].std(),  4)
        rows.append(row)

    df_m = pd.DataFrame(rows).set_index("Modelo")
    st.dataframe(
        df_m.style.background_gradient(cmap="RdYlGn",
                                        subset=["Accuracy","Precision","Recall","F1-Score"]),
        use_container_width=True,
    )

    # ── Gráfica de barras comparativa ───────────────────────
    st.markdown('<div class="sec">Comparativa visual</div>', unsafe_allow_html=True)
    met_sel = st.multiselect(
        "Métricas a graficar",
        ["Accuracy","Precision","Recall","F1-Score","MCC","Kappa"],
        default=["Accuracy","F1-Score","Precision","Recall"],
        key="met_sel",
    )

    if met_sel:
        n_m    = len(sel_models)
        n_met  = len(met_sel)
        x      = np.arange(n_m)
        width  = 0.8 / n_met
        bar_colors = [COLORS[i % len(COLORS)] for i in range(n_met)]

        fig, ax = plt.subplots(figsize=(max(8, n_m * 1.6), 4.5))
        dark_ax(ax)
        for i, met in enumerate(met_sel):
            vals = [results[nm]["metrics"][met] for nm in sel_models]
            bars = ax.bar(x + i*width - (n_met-1)*width/2, vals,
                          width=width*0.9, label=met, color=bar_colors[i], alpha=0.88)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(sel_models, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Cross-Validation boxplots ────────────────────────────
    if do_cv:
        st.markdown('<div class="sec">Distribución Cross-Validation</div>', unsafe_allow_html=True)
        cv_data  = [results[nm]["cv"] for nm in sel_models if results[nm]["cv"] is not None]
        cv_names = [nm for nm in sel_models if results[nm]["cv"] is not None]

        if cv_data:
            fig, ax = plt.subplots(figsize=(max(7, len(cv_names)*1.5), 4))
            dark_ax(ax)
            bp = ax.boxplot(cv_data, labels=cv_names, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            for patch, col in zip(bp["boxes"], [COLORS[i % 3] for i in range(len(cv_names))]):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
            ax.set_ylabel("Accuracy")
            ax.set_title("Cross-Validation Accuracy Distribution")
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ════════════════════════════════════════════════════════════
# TAB 2 ── FRONTERAS DE DECISIÓN
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="sec">Fronteras de Decisión — {feat_x} vs {feat_y}</div>',
                unsafe_allow_html=True)

    fi1 = FEAT_NAMES.index(feat_x)
    fi2 = FEAT_NAMES.index(feat_y)
    X2  = iris.data[:, [fi1, fi2]]

    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, iris.target, test_size=test_pct/100, random_state=seed, stratify=iris.target)
    sc2      = StandardScaler()
    X2_tr_sc = sc2.fit_transform(X2_tr)
    X2_te_sc = sc2.transform(X2_te)

    x_min = X2_tr_sc[:,0].min() - .6
    x_max = X2_tr_sc[:,0].max() + .6
    y_min = X2_tr_sc[:,1].min() - .6
    y_max = X2_tr_sc[:,1].max() + .6
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, mesh_res),
        np.linspace(y_min, y_max, mesh_res),
    )

    n = len(sel_models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    cmap_bg  = ListedColormap(["#2a1a4e", "#0e3333", "#3e1a1a"])
    cmap_pts = COLORS

    for idx, name in enumerate(sel_models):
        ax = axes_flat[idx]
        dark_ax(ax)

        mdl2 = ALL_MODELS[name].__class__(**ALL_MODELS[name].get_params())
        mdl2.fit(X2_tr_sc, y2_tr)
        Z = mdl2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, Z, colors="white", linewidths=0.9, alpha=0.6, levels=[0.5, 1.5])

        for ci, col in enumerate(COLORS):
            mt = y2_tr == ci
            me = y2_te == ci
            ax.scatter(X2_tr_sc[mt, 0], X2_tr_sc[mt, 1],
                       c=col, s=35, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
            ax.scatter(X2_te_sc[me, 0], X2_te_sc[me, 1],
                       c=col, s=80, marker="*", edgecolors="black", linewidths=0.5, zorder=4)

        patches = [mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i]) for i in range(3)]
        ax.legend(handles=patches, fontsize=7, loc="upper left")

        acc = results[name]["metrics"]["Accuracy"]
        ax.set_title(f"{name}\nAcc = {acc:.3f}", fontsize=9, pad=6)
        ax.set_xlabel(feat_x, fontsize=8)
        ax.set_ylabel(feat_y, fontsize=8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── PCA ────────────────────────────────────────────────
    if show_pca:
        st.markdown('<div class="sec">Proyección PCA 2D (espacio escalado)</div>',
                    unsafe_allow_html=True)
        pca   = PCA(n_components=2)
        Xp_tr = pca.fit_transform(Xtr_sc)
        Xp_te = pca.transform(Xte_sc)
        var   = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 5))
        dark_ax(ax)
        for ci, col in enumerate(COLORS):
            mt = y_tr == ci
            me = y_te == ci
            ax.scatter(Xp_tr[mt, 0], Xp_tr[mt, 1],
                       c=col, s=45, alpha=0.75, label=f"{CLASS_NAMES[ci]} train", edgecolors="white", lw=0.3)
            ax.scatter(Xp_te[me, 0], Xp_te[me, 1],
                       c=col, s=100, marker="*", label=f"{CLASS_NAMES[ci]} test",
                       edgecolors="black", lw=0.5, zorder=5)
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f} %)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f} %)")
        ax.set_title("PCA – Proyección 2D del dataset completo")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════════════════════════
# TAB 3 ── ROC & PR
# ════════════════════════════════════════════════════════════
with tab3:
    y_bin = label_binarize(y_te, classes=[0, 1, 2])

    # ── ROC ──────────────────────────────────────────────────
    st.markdown('<div class="sec">Curvas ROC – One-vs-Rest por clase</div>', unsafe_allow_html=True)

    n = len(sel_models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.5*nrows))
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, name in enumerate(sel_models):
        ax = axes_flat[idx]
        dark_ax(ax)
        for ci, col in enumerate(COLORS):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], results[name]["yprob"][:, ci])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{CLASS_NAMES[ci]} (AUC={roc_auc:.2f})")
        ax.plot([0,1], [0,1], "--", color="#555", lw=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── PR Curves ────────────────────────────────────────────
    st.markdown('<div class="sec">Curvas Precision-Recall</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.5*nrows))
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, name in enumerate(sel_models):
        ax = axes_flat[idx]
        dark_ax(ax)
        for ci, col in enumerate(COLORS):
            prec, rec, _ = precision_recall_curve(y_bin[:, ci], results[name]["yprob"][:, ci])
            ap           = average_precision_score(y_bin[:, ci], results[name]["yprob"][:, ci])
            ax.plot(rec, prec, color=col, lw=2, label=f"{CLASS_NAMES[ci]} (AP={ap:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ════════════════════════════════════════════════════════════
# TAB 4 ── MATRICES DE CONFUSIÓN
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">Matrices de Confusión (normalizada + conteos)</div>',
                unsafe_allow_html=True)

    n = len(sel_models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.2*nrows))
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, name in enumerate(sel_models):
        ax = axes_flat[idx]
        ax.set_facecolor(SURFACE)
        cm       = confusion_matrix(results[name]["yte"], results[name]["yp"])
        cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", ax=ax,
                    cmap="Blues", linewidths=0.5, linecolor=GRID,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar_kws={"shrink": 0.75})
        ax.set_title(name, color=TEXT, fontsize=9, pad=8)
        ax.set_xlabel("Predicho", color=TEXT, fontsize=8)
        ax.set_ylabel("Real",     color=TEXT, fontsize=8)
        ax.tick_params(colors=TEXT, labelsize=7)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 Reportes de clasificación completos"):
        for name, r in results.items():
            st.markdown(f"**{name}**")
            st.code(classification_report(r["yte"], r["yp"], target_names=CLASS_NAMES))

# ════════════════════════════════════════════════════════════
# TAB 5 ── ANÁLISIS AVANZADO
# ════════════════════════════════════════════════════════════
with tab5:

    # ── Feature Importance / Coeficientes ──────────────────
    if show_imp:
        st.markdown('<div class="sec">Importancia / Coeficientes de Features</div>',
                    unsafe_allow_html=True)
        imp_items = {n: r for n, r in results.items()
                     if hasattr(r["model"], "feature_importances_")
                     or hasattr(r["model"], "coef_")}

        if imp_items:
            ncols = min(2, len(imp_items))
            nrows = (len(imp_items) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 3.5*nrows))
            fig.patch.set_facecolor(BG)
            axes_flat = np.array(axes).flatten() if len(imp_items) > 1 else [axes]

            for idx, (name, r) in enumerate(imp_items.items()):
                ax = axes_flat[idx]
                dark_ax(ax)
                mdl = r["model"]
                if hasattr(mdl, "feature_importances_"):
                    imp   = mdl.feature_importances_
                    title = "Feature Importances"
                else:
                    imp   = np.mean(np.abs(mdl.coef_), axis=0)
                    title = "|Coeficientes| promedio"

                bars = ax.barh(FEAT_NAMES, imp, color=COLORS[0], alpha=0.85)
                for bar, v in zip(bars, imp):
                    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                            f"{v:.4f}", va="center", fontsize=7, color=TEXT)
                ax.set_title(f"{name} – {title}", fontsize=9)
                ax.set_xlim(0, imp.max() * 1.25)
                ax.invert_yaxis()

            for idx in range(len(imp_items), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Ningún modelo seleccionado expone importancias o coeficientes.")

    # ── Scatter Matrix (pair plot) ──────────────────────────
    st.markdown('<div class="sec">Scatter Matrix del Dataset</div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, 9))
    fig.patch.set_facecolor(BG)
    nf = len(FEAT_NAMES)
    for row in range(nf):
        for col in range(nf):
            ax = fig.add_subplot(nf, nf, row*nf + col + 1)
            dark_ax(ax)
            if row == col:
                for ci, color in enumerate(COLORS):
                    vals = df_iris[df_iris.target == ci][FEAT_NAMES[col]]
                    ax.hist(vals, bins=12, color=color, alpha=0.6, density=True)
            elif row > col:
                for ci, color in enumerate(COLORS):
                    mask = df_iris.target == ci
                    ax.scatter(df_iris[mask][FEAT_NAMES[col]],
                               df_iris[mask][FEAT_NAMES[row]],
                               c=color, s=8, alpha=0.6)
            else:
                ax.set_visible(False)
                continue
            if col == 0:
                ax.set_ylabel(FEAT_NAMES[row], fontsize=5, color=TEXT)
            if row == nf-1:
                ax.set_xlabel(FEAT_NAMES[col], fontsize=5, color=TEXT)
            ax.tick_params(labelsize=5)

    # leyenda global
    patches = [mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i]) for i in range(3)]
    fig.legend(handles=patches, loc="upper right", fontsize=8,
               facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Violin plots ────────────────────────────────────────
    st.markdown('<div class="sec">Distribución de Features por Clase</div>',
                unsafe_allow_html=True)
    feat_v = st.selectbox("Feature", FEAT_NAMES, key="viol_feat")
    fig, ax = plt.subplots(figsize=(7, 4))
    dark_ax(ax)
    data_per_class = [df_iris[df_iris.target == ci][feat_v].values for ci in range(3)]
    vp = ax.violinplot(data_per_class, positions=range(3), showmedians=True, showextrema=True)
    for i, (body, col) in enumerate(zip(vp["bodies"], COLORS)):
        body.set_facecolor(col)
        body.set_alpha(0.6)
    vp["cmedians"].set_color("white")
    vp["cbars"].set_color(GRID)
    vp["cmaxes"].set_color(GRID)
    vp["cmins"].set_color(GRID)
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel(feat_v)
    ax.set_title(f"Distribución de '{feat_v}' por especie")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Mapa de correlación ──────────────────────────────────
    st.markdown('<div class="sec">Mapa de Correlación de Features</div>',
                unsafe_allow_html=True)
    corr = df_iris[FEAT_NAMES].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    dark_ax(ax)
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax,
                cmap="coolwarm", linewidths=0.5, linecolor=GRID,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación entre features", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:.8rem;'>"
    "Iris Classifier Studio · Streamlit + scikit-learn + matplotlib"
    "</div>",
    unsafe_allow_html=True,
)
