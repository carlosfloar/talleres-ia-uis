# Talleres de Inteligencia Artificial - UIS

Material de talleres prácticos de IA/ML para estudiantes de pregrado.
Universidad Industrial de Santander.

Cada taller tiene un notebook ejecutable en Google Colab.

---

## Talleres

| # | Taller | Carpeta | Abrir en Colab |
|---|--------|---------|----------------|
| 1 | **Función de Activación** — Función escalón | `funcion_activacion/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/funcion_activacion/01_step_function.ipynb) |
| 2 | **Perceptrón** — Clasificación de frutas | `perceptron/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/perceptron/Perceptron_Frutas_Didactico.ipynb) |
| 3 | **Perceptrón 2 entradas** — Visualización animada | `perceptron_dos_entradas/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/perceptron_dos_entradas/Perceptron_Dos_Entradas_Animado.ipynb) |
| 4 | **XOR** — MLP y visualización de hiperplano | `xor/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/xor/MLP_XOR_Visualizacion_Completa_Hiperplano.ipynb) |
| 5 | **Descenso de Gradiente** — Encontrar la mejor recta | `descenso_gradiente/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/descenso_gradiente/Descenso_Gradiente_Didactico.ipynb) |
| 6 | **ADALINE** — Cancelación adaptativa de ruido (LMS/NLMS) | `adaline/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/adaline/ADALINE_LMS_Filter.ipynb) |
| 7 | **Aproximación de funciones** — Ajuste con redes neuronales | `funcion_aproximacion/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/funcion_aproximacion/Neural_Network_Function_Fitting_AVANZADO.ipynb) |
| 8 | **Deep Learning** — CNNs para diagnóstico de plantas | `deep_learning/` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfloar/talleres-ia-uis/blob/main/deep_learning/entrenamiento_cnn_tomate.ipynb) |

### Secuencia recomendada

Función de Activación → Perceptrón → Perceptrón 2 entradas → XOR (MLP) → Descenso de Gradiente → ADALINE → Aproximación de funciones → Deep Learning

Cada taller construye sobre los conceptos del anterior: funciones de activación → clasificación binaria → problemas no lineales → optimización → filtrado adaptativo → aproximación universal → redes profundas.

---

## Inicio rápido

1. Haz clic en el badge **"Open in Colab"** del taller que quieras
2. En Colab: `Entorno de ejecución → Cambiar tipo de entorno` → GPU (para Deep Learning)
3. Ejecuta las celdas en orden

---

## Contenido por taller

### 1. Función de Activación
- Función escalón (step function)
- Base para entender la neurona artificial

### 2. Perceptrón
- Clasificador binario: ¿naranja o plátano?
- Regla de aprendizaje del Perceptrón (Rosenblatt, 1958)
- Frontera de decisión y convergencia

### 3. Perceptrón 2 entradas
- Visualización animada del aprendizaje
- Frontera de decisión en 2D

### 4. XOR (MLP)
- El problema XOR: límite del Perceptrón simple
- MLP (Multi-Layer Perceptron) con capa oculta
- Visualización del hiperplano de separación en 3D

### 5. Descenso de Gradiente
- Regresión lineal con gradiente descendente
- Visualización del paisaje de error
- Efecto de la tasa de aprendizaje

### 6. ADALINE (Adaptive Linear Neuron)
- Cancelación adaptativa de ruido (ANC)
- Algoritmos LMS y NLMS (Widrow-Hoff, 1960)
- Filtrado de ruido de motor en señal de audio

### 7. Aproximación de funciones
- Redes neuronales como aproximadores universales
- Ajuste de funciones no lineales

### 8. Deep Learning
- Redes Neuronales Convolucionales (CNNs)
- Transfer Learning con ResNet18
- Diagnóstico de enfermedades en hojas de tomate (PlantVillage)
- App interactiva Streamlit (`deep_learning/app_didactica_cnn.py`)

---

## Autores

- **Ph.D, M.Sc Carlos Borrás Pinilla** — Universidad Industrial de Santander
- **M.Sc Carlos Alberto Flórez Arias** — carlosfloar@gmail.com

*Última actualización: Marzo 2026*
