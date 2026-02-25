# Deep Learning Workshop 🚀

Taller introductorio de **Deep Learning** aplicado a la detección de enfermedades en plantas.

## 📚 Contenido del taller

Este repositorio contiene todo lo necesario para aprender Deep Learning desde cero:

### 1. **Notebooks teóricos y prácticos** (`hojas/`)
- `taller_cnn_plantas_colab.ipynb` - Introducción visual a las CNNs (1 hora)
- `entrenamiento_cnn_tomate.ipynb` - Entrenamiento práctico con datos reales (~30 min)

### 2. **Aplicación interactiva**
- `app_didactica_cnn.py` - App Streamlit con visualizaciones interactivas

### 3. **Guías de ejecución**
- `GUIA_EJECUCION_COLAB.txt` - Instrucciones para Google Colab (recomendado, sin instalación)
- `GUIA_EJECUTAR_SCRIPT_PYTHON.txt` - Instrucciones para Windows local

---

## 🎯 Objetivos de aprendizaje

Al completar este taller, entenderás:

- ✅ Cómo una computadora "ve" imágenes (píxeles, matrices, filtros)
- ✅ Qué son las Redes Neuronales Convolucionales (CNNs)
- ✅ Cómo entrena una red neuronal (gradientes, backpropagation)
- ✅ Transfer Learning y modelos pre-entrenados
- ✅ Aplicación práctica: diagnosticar enfermedades en hojas de plantas

---

## 📊 Dataset

**PlantVillage Dataset**
- 54,305 imágenes de hojas etiquetadas
- 14 tipos de plantas diferentes
- 39 clases de enfermedades/salud

En este taller usamos **10 clases de tomate** para entrenar desde cero.

---

## 🚀 Inicio rápido

### Opción 1: Google Colab (Recomendado - sin instalación)

1. Abre este notebook en Colab:
   - [taller_cnn_plantas_colab.ipynb](hojas/taller_cnn_plantas_colab.ipynb)
2. Activa GPU en `Runtime → Change runtime type → GPU`
3. ¡Ejecuta las celdas!

### Opción 2: Python local (Windows)

```bash
# Clonar repositorio
git clone https://github.com/carlosfloar/deep-learning-workshop.git
cd deep-learning-workshop

# Crear ambiente virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar app Streamlit
streamlit run app_didactica_cnn.py
```

---

## 📖 Conceptos clave

### CNNs (Convolutional Neural Networks)
Redes diseñadas para procesar imágenes. Usan **filtros** que se deslizan sobre la imagen buscando patrones:
- Primeras capas detectan **bordes**
- Capas medias detectan **texturas**
- Capas profundas reconocen **objetos completos**

### Transfer Learning
Usar un modelo pre-entrenado (ResNet, VGG, etc.) como punto de partida, en lugar de entrenar desde cero. Ahorra tiempo y datos.

### Overfitting vs. Underfitting
- **Underfitting**: modelo demasiado simple, no aprende
- **Overfitting**: modelo memoriza en vez de generalizar
- **Punto óptimo**: balance entre ambos

---

## 📈 Resultados

Nuestro modelo CNN alcanza:

| Modelo | Precisión |
|--------|-----------|
| KNN | 72% |
| Árbol de Decisión | 78% |
| Regresión Logística | 82% |
| SVM | 88% |
| **CNN Personalizada** | **96.46%** |
| ResNet18 (Transfer Learning) | 97.5% |

---

## 📚 Referencias

**Paper base del taller:**
> Geetharamani, G., & Arun Pandian, J. (2019). Identification of plant leaf diseases using a nine-layer deep convolutional neural network. *Computers & Electrical Engineering*, 76, 323–338.

**Otros recursos útiles:**
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

---

## 🛠️ Tecnologías utilizadas

- **Python 3.8+**
- **PyTorch** - Framework de Deep Learning
- **Jupyter/Colab** - Notebooks interactivos
- **Streamlit** - Web app interactiva
- **Matplotlib/Plotly** - Visualización
- **Scikit-learn** - Métricas de evaluación

---

## 💬 Preguntas frecuentes

**¿Necesito GPU?**
- Para el notebook teórico: No, funciona en CPU
- Para entrenar modelos: Sí, es mucho más rápido. Google Colab T4 es gratuito ✅

**¿Necesito instalar Python?**
- No si usas Google Colab. Sí si lo haces localmente (Windows/Mac/Linux)

**¿Cuánto tiempo toma?**
- Notebook teórico: ~1 hora
- Entrenamiento práctico: ~30 min (con GPU) o ~2 horas (CPU)

---

## 📝 Licencia

Este material está disponible bajo licencia MIT.

---

## 👨‍🏫 Autor

**Prof. Carlos Floar**
Taller de Deep Learning - Universidad [Tu Universidad]

---

*Última actualización: Febrero 2026*
