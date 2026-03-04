"""
Aplicación didáctica de Deep Learning para detección de enfermedades en plantas.
Basado en el paper: Geetharamani & Arun Pandian (2019)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from scipy.signal import convolve2d
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página
st.set_page_config(
    page_title="Deep Learning para Agricultura 🌱",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .section-title {
        font-size: 2rem;
        color: #4a5568;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - Navegación
# ============================================
st.sidebar.markdown("# 🌱 Navegación")
page = st.sidebar.radio("Ir a:", [
    "🏠 Inicio",
    "🌾 El Problema", 
    "📊 El Dataset",
    "🧠 ¿Cómo ve una CNN?",
    "⚙️ La Convolución",
    "🏗️ Arquitectura",
    "🏋️ Simulador de Entrenamiento",
    "🎯 Resultados",
    "💻 Código PyTorch"
])

# ============================================
# PÁGINA: INICIO
# ============================================
if page == "🏠 Inicio":
    st.markdown('<h1 class="main-title">Deep Learning para Agricultura</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.3rem; color: #666; margin: 20px 0;">
        Detectando enfermedades en hojas de plantas con Redes Neuronales Convolucionales
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📸 Imágenes", "54,305")
    with col2:
        st.metric("🏷️ Clases", "39")
    with col3:
        st.metric("🎯 Accuracy", "96.46%")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Objetivo de este tutorial</h3>
        <p style="font-size: 1.1rem;">
            Entender <b>cómo las computadoras pueden "ver"</b> y diagnosticar enfermedades en plantas, 
            paso a paso, sin necesidad de ser experto en programación.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Lo que aprenderás:")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        - 🧠 Qué son las CNNs y cómo funcionan
        - ⚙️ Qué es una convolución (visualmente)
        - 🏗️ Arquitectura de una red de 9 capas
        """)
    with cols[1]:
        st.markdown("""
        - 🏋️ Cómo se entrena una red neuronal
        - 📊 Cómo evaluar el rendimiento
        - 💻 Implementación en PyTorch
        """)

# ============================================
# PÁGINA: EL PROBLEMA
# ============================================
elif page == "🌾 El Problema":
    st.markdown('<h1 class="section-title">🌾 El Problema</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Las enfermedades en cultivos causan **pérdidas millonarias** anuales. 
    La detección temprana puede salvar cosechas enteras.
    """)
    
    # Crear visualización de hojas
    def create_leaf_visual(state='healthy'):
        size = 150
        img = np.ones((size, size, 3))
        Y, X = np.ogrid[:size, :size]
        center = size // 2
        ellipse = ((X - center)**2 / 4500 + (Y - center)**2 / 2800) < 1
        
        if state == 'healthy':
            img[ellipse] = [0.2, 0.7, 0.2]
        elif state == 'mild':
            img[ellipse] = [0.5, 0.7, 0.3]
            spots = ((X - center-20)**2 + (Y - center+15)**2) < 120
            img[spots & ellipse] = [0.6, 0.5, 0.2]
        else:
            img[ellipse] = [0.6, 0.5, 0.2]
            spots = ((X - center-15)**2 + (Y - center-8)**2) < 400
            img[spots & ellipse] = [0.4, 0.3, 0.1]
        
        img[~ellipse] = [0.95, 0.95, 0.95]
        return img
    
    col1, col2, col3 = st.columns(3)
    
    states = [
        ('healthy', '✅ Sana', '#27ae60', 'Sin tratamiento necesario'),
        ('mild', '⚠️ Enfermedad Temprana', '#f39c12', 'Tratamiento temprano efectivo'),
        ('severe', '❌ Enfermedad Avanzada', '#e74c3c', 'Pérdida probable del cultivo')
    ]
    
    for col, (state, title, color, desc) in zip([col1, col2, col3], states):
        with col:
            leaf = create_leaf_visual(state)
            st.image(leaf, caption=title, use_column_width=True)
            
            # Gráfico de impacto
            if state == 'healthy':
                impact_data = {'Detección': 20, 'Costo': 10, 'Pérdida': 0}
            elif state == 'mild':
                impact_data = {'Detección': 60, 'Costo': 40, 'Pérdida': 20}
            else:
                impact_data = {'Detección': 95, 'Costo': 90, 'Pérdida': 85}
            
            fig = px.bar(
                x=list(impact_data.keys()),
                y=list(impact_data.values()),
                color=list(impact_data.values()),
                color_continuous_scale=[color],
                range_y=[0, 100],
                title="Impacto (%)",
                height=200
            )
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(desc)
    
    st.info("💡 **Insight clave:** Detectar enfermedades **temprano** reduce costos y pérdidas drásticamente.")

# ============================================
# PÁGINA: DATASET
# ============================================
elif page == "📊 El Dataset":
    st.markdown('<h1 class="section-title">📊 El Dataset: PlantVillage</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    El paper usa el dataset **PlantVillage** con:
    - **54,305 imágenes** de hojas de plantas
    - **39 clases** diferentes (tipos de plantas y enfermedades)
    - **14 tipos de plantas**: Manzana, Tomate, Uva, Papa, Maíz, etc.
    """)
    
    # Datos del dataset
    PLANT_CLASSES = {
        'Manzana': 4, 'Arándano': 1, 'Cereza': 2, 'Maíz': 4,
        'Uva': 4, 'Naranja': 1, 'Durazno': 2, 'Pimiento': 2,
        'Papa': 3, 'Frambuesa': 1, 'Soya': 1, 'Calabaza': 1,
        'Fresa': 2, 'Tomate': 10, 'Fondo': 1
    }
    
    fig = px.bar(
        x=list(PLANT_CLASSES.values()),
        y=list(PLANT_CLASSES.keys()),
        orientation='h',
        color=list(PLANT_CLASSES.values()),
        color_continuous_scale='viridis',
        labels={'x': 'Número de clases (estados de salud)', 'y': 'Planta'},
        title='Distribución de clases en PlantVillage'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de clases", sum(PLANT_CLASSES.values()))
    col2.metric("Planta con más clases", "Tomate (10)")
    col3.metric("Imágenes totales", "54,305")

# ============================================
# PÁGINA: ¿CÓMO VE UNA CNN?
# ============================================
elif page == "🧠 ¿Cómo ve una CNN?":
    st.markdown('<h1 class="section-title">🧠 ¿Cómo ve una computadora?</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Analogía: De los ojos humanos a las redes neuronales
    
    Cuando **tú** ves una manzana:
    1. Ojos reciben la imagen
    2. Cerebro detecta bordes, colores, texturas
    3. Combina pistas y reconoce el objeto
    
    Una **CNN** hace exactamente lo mismo, pero en capas:
    """)
    
    # Visualización del proceso
    size = 80
    Y, X = np.ogrid[:size, :size]
    center = size // 2
    
    # Crear imagen de hoja
    original = np.zeros((size, size))
    ellipse = ((X - center)**2 / 1200 + (Y - center)**2 / 800) < 1
    original[ellipse] = 0.8
    spot = ((X - center - 15)**2 + (Y - center + 10)**2) < 100
    original[spot] = 0.3
    
    # Procesar
    edges_v = np.abs(np.gradient(original, axis=1))
    edges_h = np.abs(np.gradient(original, axis=0))
    edges = np.sqrt(edges_v**2 + edges_h**2)
    texture = np.abs(np.gradient(edges)[0]) + np.abs(np.gradient(edges)[1])
    
    # Mostrar proceso
    cols = st.columns(4)
    images = [
        (original, 'Greens', '1️⃣ Original', 'Imagen de entrada'),
        (edges, 'hot', '2️⃣ Bordes', 'Capas 1-2: Detectan bordes'),
        (texture, 'viridis', '3️⃣ Texturas', 'Capas 3-4: Detectan patrones'),
        (original > 0.5, 'plasma', '4️⃣ Decisión', 'Capas 5+: Clasificación')
    ]
    
    for col, (img, cmap, title, desc) in zip(cols, images):
        with col:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap=cmap)
            ax.axis('off')
            ax.set_title(title, fontsize=12, fontweight='bold')
            st.pyplot(fig)
            st.caption(desc)
    
    st.markdown("""
    ### 📚 Jerarquía de características:
    
    | Capa | Qué detecta | Ejemplo en hojas |
    |------|-------------|------------------|
    | **1-2** | Bordes simples | Líneas, curvas, colores básicos |
    | **3-4** | Texturas y formas | Venas de la hoja, manchas circulares |
    | **5+** | Objetos completos | Tipo específico de enfermedad |
    """)

# ============================================
# PÁGINA: LA CONVOLUCIÓN
# ============================================
elif page == "⚙️ La Convolución":
    st.markdown('<h1 class="section-title">⚙️ La Convolución Visual</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## ¿Qué es una convolución?
    
    Es una **ventana deslizante** que pasa por la imagen buscando patrones específicos.
    """
    )
    
    # Selector de filtro
    filter_type = st.selectbox(
        "Selecciona un tipo de filtro:",
        ["Detector de Bordes", "Suavizado", "Detector de Esquinas"]
    )
    
    # Crear imagen de ejemplo
    img = np.zeros((10, 10))
    img[3:7, 3:7] = 1
    
    # Definir filtro
    if filter_type == "Detector de Bordes":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        desc = "Resalta cambios bruscos (bordes)"
    elif filter_type == "Suavizado":
        kernel = np.ones((3, 3)) / 9
        desc = "Reduce ruido, suaviza la imagen"
    else:
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        desc = "Encuentra puntos donde convergen bordes"
    
    result = convolve2d(img, kernel, mode='same')
    
    # Mostrar
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title('IMAGEN ORIGINAL', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=8)
        ax.set_title(f'FILTRO\n{filter_type}', fontweight='bold')
        ax.axis('off')
        # Números del kernel
        for i in range(3):
            for j in range(3):
                color = 'white' if abs(kernel[i,j]) > 3 else 'black'
                ax.text(j, i, f'{kernel[i,j]:.1f}', ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=12)
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(result, cmap='hot')
        ax.set_title('RESULTADO\n(Feature Map)', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
    
    st.success(f"💡 **{filter_type}:** {desc}")

# ============================================
# PÁGINA: ARQUITECTURA
# ============================================
elif page == "🏗️ Arquitectura":
    st.markdown('<h1 class="section-title">🏗️ Arquitectura del Modelo</h1>', unsafe_allow_html=True)
    
    st.markdown("La CNN de 9 capas del paper:")
    
    # Crear diagrama
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Título
    ax.text(11, 11.5, 'Arquitectura PlantDiseaseCNN (9 Capas)', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Colores
    colors = {
        'input': '#90EE90', 'conv': '#87CEEB', 'pool': '#FFD700',
        'dense': '#FF6B6B', 'output': '#98FB98'
    }
    
    # Capas
    layers = [
        ('INPUT\n128×128×3', 0.5, 5, 1.5, 4, colors['input']),
        ('CONV1\n32 filtros\n3×3', 2.5, 5, 1.2, 4, colors['conv']),
        ('POOL\n2×2', 4, 5, 0.8, 4, colors['pool']),
        ('CONV2\n64 filtros', 5.2, 5, 1.2, 4, colors['conv']),
        ('POOL', 6.7, 5, 0.8, 4, colors['pool']),
        ('CONV3\n128', 7.8, 5, 1, 4, colors['conv']),
        ('POOL', 9, 5, 0.7, 4, colors['pool']),
        ('CONV4\n256', 9.9, 5, 1, 4, colors['conv']),
        ('POOL', 11.1, 5, 0.7, 4, colors['pool']),
        ('CONV5\n512', 12, 5, 1, 4, colors['conv']),
        ('POOL', 13.2, 5, 0.7, 4, colors['pool']),
        ('CONV6\n512', 14.1, 5, 1, 4, colors['conv']),
        ('POOL', 15.3, 5, 0.7, 4, colors['pool']),
        ('FLATTEN\n2048', 16.2, 5, 1, 4, '#FFA500'),
    ]
    
    for name, x, y, w, h, color in layers:
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', 
                fontsize=8, fontweight='bold')
    
    # Flechas
    for i in range(len(layers)-1):
        x1 = layers[i][1] + layers[i][3]
        y1 = layers[i][2] + layers[i][4]/2
        x2 = layers[i+1][1]
        y2 = layers[i+1][2] + layers[i+1][4]/2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Capas densas abajo
    dense = [
        ('DENSE 1\n2048→512\n+ Dropout 0.5', 5, 1.5, 3, 2, colors['dense']),
        ('DENSE 2\n512→512\n+ Dropout 0.5', 9, 1.5, 3, 2, colors['dense']),
        ('OUTPUT\n512→39\nSoftmax', 13.5, 1.5, 2.5, 2, colors['output']),
    ]
    
    for name, x, y, w, h, color in dense:
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Flechas entre densas
    ax.annotate('', xy=(9, 2.5), xytext=(8, 2.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    ax.annotate('', xy=(13.5, 2.5), xytext=(12, 2.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    # Flecha de flatten a densas
    ax.annotate('', xy=(6.5, 3.5), xytext=(16.7, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='darkred', ls='--'))
    ax.text(17.5, 4, 'FLATTEN', fontsize=10, color='darkred', fontweight='bold')
    
    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], label='Entrada'),
        mpatches.Patch(facecolor=colors['conv'], label='Convolucional'),
        mpatches.Patch(facecolor=colors['pool'], label='Pooling'),
        mpatches.Patch(facecolor=colors['dense'], label='Densa'),
        mpatches.Patch(facecolor=colors['output'], label='Salida')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Estadísticas
    stats = """📊 ESTADÍSTICAS:
• Parámetros: ~4.6M
• Capas conv: 6
• Capas pool: 6
• Capas densas: 2
• Salida: 39 clases"""
    ax.text(0.5, 0.5, stats, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    st.pyplot(fig)
    
    st.info("💡 Observa cómo la imagen se hace más 'profunda' (más canales) pero más pequeña espacialmente.")

# ============================================
# PÁGINA: SIMULADOR DE ENTRENAMIENTO
# ============================================
elif page == "🏋️ Simulador de Entrenamiento":
    st.markdown('<h1 class="section-title">🏋️ Simulador de Entrenamiento</h1>', unsafe_allow_html=True)
    
    st.markdown("Ajusta los hiperparámetros y observa cómo afecta el aprendizaje:")
    
    # Controles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    with col2:
        batch = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=128)
    with col3:
        dropout = st.slider("Dropout", 0.0, 0.8, 0.5, 0.1)
    with col4:
        epochs = st.slider("Épocas", 20, 100, 50, 10)
    
    # Simular
    np.random.seed(42)
    lr_eff = 1 - abs(np.log10(lr) + 3) * 0.2
    max_acc = min(0.98, 0.96 * lr_eff)
    
    x = np.arange(epochs)
    train_acc = max_acc * (1 - np.exp(-x / (epochs/6))) + np.random.randn(epochs) * 0.01
    overfitting = max(0, 0.1 - dropout * 0.15)
    val_acc = train_acc - overfitting + np.random.randn(epochs) * 0.01
    val_acc = np.clip(val_acc, 0, 1)
    
    # Gráficas
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Pérdida'))
    
    fig.add_trace(go.Scatter(x=x, y=train_acc, mode='lines', name='Train', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=val_acc, mode='lines', name='Val', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=2.5*np.exp(-train_acc*2), mode='lines', name='Train Loss', 
                            line=dict(color='blue'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=2.8*np.exp(-val_acc*2), mode='lines', name='Val Loss', 
                            line=dict(color='green'), showlegend=False), row=1, col=2)
    
    fig.update_layout(height=400, title_text=f"Entrenamiento (LR={lr}, Dropout={dropout})")
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis
    gap = train_acc[-1] - val_acc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Train Accuracy", f"{train_acc[-1]:.1%}")
    col2.metric("Val Accuracy", f"{val_acc[-1]:.1%}")
    col3.metric("Gap", f"{gap:.1%}", delta=f"{'Overfitting' if gap > 0.08 else 'OK'}")
    
    if gap > 0.08:
        st.error("⚠️ Overfitting detectado. Intenta aumentar el dropout.")
    elif val_acc[-1] < 0.75:
        st.warning("⚠️ Underfitting. Intenta más épocas o mayor learning rate.")
    else:
        st.success("✅ ¡Buen balance entre bias y variance!")

# ============================================
# PÁGINA: RESULTADOS
# ============================================
elif page == "🎯 Resultados":
    st.markdown('<h1 class="section-title">🎯 Resultados del Paper</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparación de modelos
        models = ['K-NN', 'Decision Tree', 'Logistic Reg', 'SVM', 'Deep CNN']
        accuracy = [0.72, 0.78, 0.82, 0.88, 0.9646]
        colors = ['gray']*4 + ['#e74c3c']
        
        fig = px.bar(x=models, y=accuracy, color=accuracy, 
                    color_continuous_scale=['gray', '#e74c3c'],
                    labels={'x': 'Modelo', 'y': 'Accuracy'},
                    title='Comparación de modelos',
                    text=[f'{a:.1%}' for a in accuracy])
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Velocidad', 'Escalabilidad']
        svm_scores = [0.88, 0.87, 0.86, 0.865, 0.4, 0.3]
        cnn_scores = [0.9646, 0.96, 0.95, 0.955, 0.8, 0.9]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=svm_scores + [svm_scores[0]], 
                                     theta=categories + [categories[0]],
                                     fill='toself', name='SVM'))
        fig.add_trace(go.Scatterpolar(r=cnn_scores + [cnn_scores[0]], 
                                     theta=categories + [categories[0]],
                                     fill='toself', name='Deep CNN'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         showlegend=True, title='Radar de métricas', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 📈 Conclusiones:
    - La CNN supera a **todos** los métodos tradicionales
    - Mejora de **8.46 puntos** porcentuales sobre SVM (88% → 96.46%)
    - El dropout de 0.5 previno efectivamente el overfitting
    """)

# ============================================
# PÁGINA: CÓDIGO
# ============================================
elif page == "💻 Código PyTorch":
    st.markdown('<h1 class="section-title">💻 Implementación en PyTorch</h1>', unsafe_allow_html=True)
    
    st.markdown("### Clase del modelo:")
    code = '''
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=39, dropout_rate=0.5):
        super().__init__()
        
        # 6 Bloques convolucionales
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

# Crear modelo
model = PlantDiseaseCNN(num_classes=39, dropout_rate=0.5)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
'''
    st.code(code, language='python')
    
    st.markdown("### Bucle de entrenamiento:")
    train_code = '''
# Configuración
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Entrenamiento
for epoch in range(100):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validación
    model.eval()
    # ... calcular accuracy ...
    
    scheduler.step()
'''
    st.code(train_code, language='python')

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Tutorial basado en: Geetharamani & Arun Pandian (2019) - Computers and Electrical Engineering")
