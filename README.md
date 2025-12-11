# 🚀 SISTEMA DE CLASIFICACIÓN PROFESIONAL CON PyTorch

Entrenamiento de Redes Neuronales Profundas para Clasificación de Datos

---

## 📋 Descripción General

Este proyecto implementa un **sistema profesional de clasificación** basado en redes neuronales artificiales (MLP - Multi-Layer Perceptron) utilizando PyTorch. Permite entrenar modelos de deep learning con tus propios datos, evaluarlos con métricas estándar de la industria y hacer predicciones sobre nuevos datos.

### Características Principales

✅ **Arquitectura Flexible**: Capa ocultas personalizables
✅ **Regularización Avanzada**: BatchNorm, Dropout y Early Stopping
✅ **Preprocesamiento Automático**: Normalización y codificación de categorías
✅ **Visualizaciones Profesionales**: Matrices de confusión e historial de entrenamiento
✅ **Persistencia de Modelos**: Guarda y carga modelos entrenados
✅ **Soporte GPU/CPU**: Detección automática de dispositivo
✅ **Métricas Completas**: Accuracy, F1-Score, Reporte de clasificación

---

## 🔧 Requisitos

- **Python**: 3.10 (64-bit) ⚠️ *Importante: Se requiere Python 64-bit*
- **Sistema Operativo**: Windows, macOS o Linux

### Dependencias Principales

```
torch==2.0.1+cpu
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
matplotlib==3.10.7
seaborn==0.13.2
joblib==1.5.2
```

---

## 📦 Instalación

### 1. Crear Entorno Virtual con Python 3.10

```powershell
# En Windows PowerShell
python -m venv venv_torch
```

### 2. Activar el Entorno Virtual

```powershell
# En Windows PowerShell
.\venv_torch\Scripts\Activate.ps1
```

### 3. Instalar Dependencias

```powershell
pip install torch==2.0.1+cpu pandas==2.3.3 numpy==2.2.6 scikit-learn==1.7.2 matplotlib==3.10.7 seaborn==0.13.2 joblib==1.5.2
```

---

## 📂 Estructura del Proyecto

```
proyectos 2/
├── clasificador.py              # Archivo principal con todas las clases
├── clasificador_profesional.py   # Copia del código fuente
├── venv_torch/                   # Entorno virtual (NO incluir en repositorio)
├── modelos_guardados/            # Modelos entrenados (se crea automáticamente)
├── resultados/                   # Gráficas y visualizaciones (se crea automáticamente)
└── README.md                     # Este archivo
```

---

## 🏗️ Estructura del Código

### Clase `Config`
Gestiona todos los parámetros de configuración del modelo:

```python
class Config:
    # Arquitectura
    hidden_layers = [64, 64]      # Neuronas en capas ocultas
    dropout_rate = 0.2             # Tasa de dropout
    
    # Entrenamiento
    epochs = 50                    # Épocas de entrenamiento
    learning_rate = 0.001          # Tasa de aprendizaje
    batch_size = 32                # Tamaño del lote
    validation_split = 0.2         # % de datos para validación
```

### Clase `FlexibleMLP`
Red Neuronal Multi-Capa con arquitectura personalizable:

- **Capas Dinámicas**: Se adapta a cualquier número de características
- **BatchNormalization**: Normaliza activaciones entre capas
- **Activación ReLU**: Introducida entre capas ocultas
- **Dropout**: Regularización para evitar sobreajuste

```
Entrada → Linear → BatchNorm1d → ReLU → Dropout → 
          Linear → BatchNorm1d → ReLU → Dropout → 
          Linear (salida)
```

### Clase `ClasificadorProfesional`
Orquestador principal que maneja todo el proceso ML:

#### Métodos Principales:

| Método | Descripción |
|--------|-------------|
| `cargar_datos_csv()` | Carga datos desde archivo CSV |
| `preparar_datos()` | Normaliza y divide datos (80/20) |
| `crear_modelo()` | Construye la red neuronal |
| `entrenar()` | Entrena el modelo con early stopping |
| `evaluar()` | Calcula métricas y genera visualizaciones |
| `plot_historial()` | Grafica pérdida y accuracy |
| `guardar_modelo()` | Persiste el modelo entrenado |
| `cargar_modelo()` | Carga un modelo previamente guardado |
| `predecir()` | Hace predicciones en nuevos datos |

---

## 💻 Ejemplos de Uso

### Opción 1: Ejecutar con Datos Sintéticos (Demo)

```python
from clasificador import ClasificadorProfesional, Config, ejemplo_uso_datos_sinteticos

# Ejecuta la demostración completa
ejemplo_uso_datos_sinteticos()
```

**Salida esperada**:
- Crea 2000 datos sintéticos con 20 características y 3 clases
- Entrena la red durante hasta 50 épocas
- Muestra métricas: Accuracy, F1-Score, Matriz de Confusión
- Genera gráficas en carpeta `resultados/`
- Guarda el modelo en carpeta `modelos_guardados/`

---

### Opción 2: Usar con Tus Propios Datos CSV

#### Paso 1: Preparar el CSV
Tu archivo CSV debe tener:
- Una columna con la **variable objetivo** (lo que quieres predecir)
- Las demás columnas como **características** (features)

Ejemplo `datos.csv`:
```
feature1,feature2,feature3,...,target
1.2,0.5,2.1,...,A
2.1,1.3,0.8,...,B
0.9,2.2,1.5,...,A
```

#### Paso 2: Entrenar el Modelo

```python
from clasificador import ClasificadorProfesional, Config
import numpy as np

# Inicializa el clasificador
config = Config()
config.epochs = 100          # Puedes personalizar parámetros
config.learning_rate = 0.0005

clasificador = ClasificadorProfesional(config)

# Carga los datos
X, y = clasificador.cargar_datos_csv(
    ruta_csv="ruta/a/tu/datos.csv",
    columna_objetivo="nombre_columna_objetivo",
    columnas_excluir=["id", "nombre"]  # Opcional: columnas a ignorar
)

# Prepara los datos
X_train, X_val, y_train, y_val = clasificador.preparar_datos(X, y)

# Crea el modelo
clasificador.crear_modelo(
    input_size=X.shape[1],
    output_size=len(np.unique(y))
)

# Entrena
clasificador.entrenar(X_train, y_train, X_val, y_val)

# Evalúa
accuracy, f1, cm = clasificador.evaluar(X_val, y_val)

# Guarda visualizaciones
clasificador.plot_historial()

# Guarda el modelo
ruta_modelo = clasificador.guardar_modelo("mi_modelo")
```

---

### Opción 3: Hacer Predicciones con Modelo Guardado

```python
from clasificador import ClasificadorProfesional

# Carga un modelo guardado
clasificador = ClasificadorProfesional()
clasificador.cargar_modelo("modelos_guardados/mi_modelo_20251211_120000.pkl")

# Haz predicciones
X_nuevo = [[1.2, 0.5, 2.1, ...]]  # Array con nuevos datos
predicciones, probabilidades = clasificador.predecir(X_nuevo)

print(f"Predicción: {predicciones}")
print(f"Probabilidades: {probabilidades}")
```

---

## 📊 Salida del Programa

### Durante Inicialización
```
============================================================
🚀 CLASIFICADOR PROFESIONAL INICIALIZADO
============================================================
📱 Dispositivo: cpu
📂 Modelos guardados en: modelos_guardados
📊 Resultados guardados en: resultados
```

### Durante Entrenamiento
```
============================================================
🎯 INICIANDO ENTRENAMIENTO
============================================================

Época 001/050 | Loss Entreno: 1.0923 | Loss Val: 1.0234 | Precisión Val: 45.32%
Época 005/050 | Loss Entreno: 0.6234 | Loss Val: 0.5892 | Precisión Val: 78.15%
...
```

### Después de Entrenar
```
============================================================
📊 EVALUACIÓN DEL MODELO
============================================================

🎯 Precisión (Accuracy): 87.50%
📈 F1-Score: 0.8645

📋 REPORTE DETALLADO POR CLASE:
---
              precision    recall  f1-score   support
       Clase A     0.8900   0.8700   0.8800       120
       Clase B     0.8400   0.8600   0.8500       100
...
```

---

## 🔍 Parámetros de Configuración

### Arquitectura de la Red

| Parámetro | Rango Recomendado | Efecto |
|-----------|-------------------|--------|
| `hidden_layers` | `[32, 64, 128]` | Lista de neuronas por capa |
| `dropout_rate` | `0.1 - 0.5` | Mayor = más regularización |

### Entrenamiento

| Parámetro | Rango Recomendado | Efecto |
|-----------|-------------------|--------|
| `epochs` | `50 - 200` | Más épocas = más tiempo pero mejor aprendizaje |
| `learning_rate` | `0.0001 - 0.01` | Más alto = aprendizaje más rápido pero inestable |
| `batch_size` | `16 - 128` | Más grande = más memoria pero más rápido |

---

## 🎯 Técnicas Avanzadas Implementadas

### Early Stopping
Detiene el entrenamiento si no hay mejora durante 10 épocas consecutivas:
```python
if patience_counter >= 10:
    print("⚠ Early stopping activado")
    break
```

### Learning Rate Scheduler
Reduce la tasa de aprendizaje si la pérdida no mejora:
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### Model Checkpointing
Guarda automáticamente el mejor modelo durante el entrenamiento:
```python
if val_loss < best_val_loss:
    self.best_model_state = self.model.state_dict().copy()
```

---

## 📈 Métricas de Evaluación

### Accuracy (Precisión)
Porcentaje de predicciones correctas:
$$\text{Accuracy} = \frac{\text{Predicciones Correctas}}{\text{Total de Predicciones}}$$

### F1-Score
Promedio ponderado de precisión y recall:
$$F1 = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}$$

### Matriz de Confusión
Tabla que muestra verdaderos positivos, falsos positivos, etc.

---

## 🐛 Solución de Problemas

### Error: `ModuleNotFoundError: No module named 'torch'`

**Solución**: Asegúrate de activar el entorno virtual:
```powershell
.\venv_torch\Scripts\Activate.ps1
```

### Error: `Python 32-bit not compatible with PyTorch`

**Solución**: Necesitas Python 3.10 64-bit. Descárgalo de [python.org](https://www.python.org/downloads/)

### Error: `RuntimeError: CUDA out of memory`

**Solución**: Reduce `batch_size` en la configuración o usa CPU en lugar de GPU.

### Las gráficas no se muestran

**Solución**: Las gráficas se guardan automáticamente en la carpeta `resultados/`. Ábrelas con un explorador de archivos.

---

## 📚 Conceptos Clave

### MLP (Multi-Layer Perceptron)
Red neuronal feedforward con múltiples capas ocultas que aprende representaciones complejas.

### BatchNormalization
Normaliza las entradas de cada capa para acelerar el entrenamiento y mejorar la estabilidad.

### Dropout
Desactiva aleatoriamente neuronas durante el entrenamiento para evitar sobreajuste.

### CrossEntropyLoss
Función de pérdida estándar para problemas de clasificación multiclase.

### Adam Optimizer
Optimizador adaptativo que combina ventajas de AdaGrad y RMSprop.

---

## 📄 Estructura de Archivos Guardados

### Modelos Guardados
```
modelos_guardados/
├── modelo_ejemplo_20251211_120000.pkl
├── mi_modelo_20251211_150530.pkl
└── ...
```

Contienen:
- Pesos del modelo (`model_state_dict`)
- Arquitectura (`model_architecture`)
- Normalizador (`scaler`)
- Codificador de etiquetas (`label_encoder`)
- Historial de entrenamiento (`history`)

### Resultados
```
resultados/
├── confusion_matrix_20251211_120000.png
├── training_history_20251211_120000.png
└── ...
```

---

## 🚀 Próximos Pasos

1. **Ajusta los hiperparámetros** en la clase `Config` según tu dataset
2. **Experimenta con diferentes arquitecturas** modificando `hidden_layers`
3. **Aumenta el volumen de datos** para mejor generalización
4. **Valida con datos nuevos** usando el método `predecir()`
5. **Guarda modelos prometedores** para reutilizarlos después

---

## 📞 Información de Contacto

**Creado**: Noviembre 2025
**Versión**: 1.0
**Python**: 3.10+
**PyTorch**: 2.0.1

---

## 📖 Referencias

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [scikit-learn Guide](https://scikit-learn.org/stable/)
- [Neural Networks Basics](https://en.wikipedia.org/wiki/Artificial_neural_network)

---

**¡Disfruta entrenando tus modelos!** 🎉
