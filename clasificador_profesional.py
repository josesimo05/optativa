"""
=====================================
SISTEMA DE CLASIFICACIÓN PROFESIONAL
Entrenamiento de Redes Neuronales con Datos Reales
=====================================

Este programa puede:
- Cargar datos desde archivos CSV
- Entrenar modelos con tus propios datos
- Guardar y cargar modelos entrenados
- Hacer predicciones sobre nuevos datos
- Mostrar métricas profesionales
- Generar visualizaciones
"""
# Su nombre
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os
from datetime import datetime
#13/11/2025##################################################################################################

# ========================================
# CONFIGURACIÓN PRINCIPAL
# ========================================

class Config:
    """
    Configuración del modelo - Puedo modificar estos valores según mis necesidades
    """
    # Arquitectura de la red
    hidden_layers = [64, 64]  # Número de neuronas en cada capa oculta
    dropout_rate = 0.2  # Tasa de dropout para evitar sobreajuste
    
    # Entrenamiento
    epochs = 50  # Número de épocas de entrenamiento
    learning_rate = 0.001  # Tasa de aprendizaje
    batch_size = 32  # Tamaño del lote
    validation_split = 0.2  # Porcentaje de datos para validación
    
    # Rutas de archivos
    models_dir = "modelos_guardados"  # Carpeta donde guardo los modelos
    results_dir = "resultados"  # Carpeta donde guardo las visualizaciones
    
    # Opciones
    use_gpu = True  # Usar GPU si está disponible
    random_seed = 42  # Semilla para reproducibilidad


# ========================================
# CLASE DE RED NEURONAL FLEXIBLE
# ========================================

class FlexibleMLP(nn.Module):
    """
    Red Neuronal Multi-Capa (MLP) flexible que se adapta a cualquier dataset
    """
    
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.2):
        """
        Inicializo la red neuronal con arquitectura personalizable
        
        Args:
            input_size: Número de características de entrada
            hidden_layers: Lista con el número de neuronas en cada capa oculta
            output_size: Número de clases de salida
            dropout_rate: Tasa de dropout para regularización
        """
        super(FlexibleMLP, self).__init__()
        
        # Construyo las capas dinámicamente
        layers = []
        prev_size = input_size
        
        # Añado cada capa oculta con BatchNorm, ReLU y Dropout
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Normalización por lotes
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout para evitar sobreajuste
            prev_size = hidden_size
        
        # Capa de salida (sin activación porque uso CrossEntropyLoss)
        layers.append(nn.Linear(prev_size, output_size))
        
        # Creo el modelo secuencial
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Paso hacia adelante de la red"""
        return self.network(x)
#######################################################20/11/2024

# ========================================
# CLASE PRINCIPAL DEL CLASIFICADOR
# ========================================

class ClasificadorProfesional:
    """
    Clase principal que maneja todo el proceso de clasificación
    """
    
    def __init__(self, config=Config()):
        """
        Inicializo el clasificador con la configuración dada
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = self._get_device()
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Creo las carpetas necesarias
        os.makedirs(config.models_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
#############################################################################20/11/2024
        print("=" * 60)
        print("🚀 CLASIFICADOR PROFESIONAL INICIALIZADO")
        print("=" * 60)
        print(f"📱 Dispositivo: {self.device}")
        print(f"📂 Modelos guardados en: {config.models_dir}")
        print(f"📊 Resultados guardados en: {config.results_dir}")
        print()
    
    def _get_device(self):
        """
        Determino si uso GPU o CPU para el entrenamiento
        """
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def cargar_datos_csv(self, ruta_csv, columna_objetivo, columnas_excluir=None):
        """
        Cargo datos desde un archivo CSV
        
        Args:
            ruta_csv: Ruta al archivo CSV
            columna_objetivo: Nombre de la columna con las etiquetas
            columnas_excluir: Lista de columnas a excluir (opcional)
        
        Returns:
            X, y: Características y etiquetas
        """
        print("=" * 60)
        print("📁 CARGANDO DATOS DESDE CSV")
        print("=" * 60)
        
        # Leo el archivo CSV
        df = pd.read_csv(ruta_csv)
        print(f"✓ Archivo cargado: {ruta_csv}")
        print(f"✓ Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"✓ Columnas: {list(df.columns)}")
        print()
        
        # Separo las características (X) de las etiquetas (y)
        if columna_objetivo not in df.columns:
            raise ValueError(f"La columna '{columna_objetivo}' no existe en el CSV")
#################################################0##########################27/11/2025        
        y = df[columna_objetivo].values
        
        columnas_a_eliminar = [columna_objetivo]
        if columnas_excluir:
            columnas_a_eliminar.extend(columnas_excluir)
        
        X = df.drop(columns=columnas_a_eliminar, errors='ignore')
        
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"⚠ Convirtiendo columna categórica '{col}' a numérica...")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.values.astype(np.float32)
        
        print(f"✓ Características: {X.shape[1]} columnas")
        print(f"✓ Clases únicas: {np.unique(y)}")
        print()
        
        return X, y
######################################################################333###4/12/    
    def preparar_datos(self, X, y):
        """
        Preparo los datos para el entrenamiento
        """
        print("=" * 60)
        print("🔧 PREPARANDO DATOS")
        print("=" * 60)
######################################################################333###4/12/    
        
        y = self.label_encoder.fit_transform(y)
        print(f"✓ Etiquetas codificadas: {self.label_encoder.classes_}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=self.config.random_seed,
            stratify=y  # Mantengo la proporción de clases
        )
        
        print(f"✓ Datos de entrenamiento: {X_train.shape[0]} ejemplos")
        print(f"✓ Datos de validación: {X_val.shape[0]} ejemplos")
        
        # Normalizo los datos
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        print("✓ Datos normalizados (StandardScaler)")
        
        # Convierto a tensores de PyTorch
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        print("✓ Datos convertidos a tensores de PyTorch")
        print()
        
        return X_train, X_val, y_train, y_val
    
    def crear_modelo(self, input_size, output_size):
        """
        Creo la red neuronal con la arquitectura definida
        """
        print("=" * 60)
        print("🏗 CONSTRUYENDO RED NEURONAL")
        print("=" * 60)
        
        self.model = FlexibleMLP(
            input_size=input_size,
            hidden_layers=self.config.hidden_layers,
            output_size=output_size,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Cuento los parámetros del modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Arquitectura: {input_size} → {' → '.join(map(str, self.config.hidden_layers))} → {output_size}")
        print(f"✓ Parámetros totales: {total_params:,}")
        print(f"✓ Parámetros entrenables: {trainable_params:,}")
        print()
    
    def entrenar(self, X_train, y_train, X_val, y_val):
        """
        Entreno el modelo con los datos proporcionados
        """
        print("=" * 60)
        print("🎯 INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        print()
        
        # Defino el optimizador y la función de pérdida
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Creo un scheduler para ajustar el learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        # Bucle de entrenamiento
        for epoch in range(self.config.epochs):
            # Modo entrenamiento
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_train)
            train_loss = criterion(outputs, y_train)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Evaluación en validación
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_predictions = val_outputs.argmax(dim=1)
                val_accuracy = (val_predictions == y_val).float().mean().item()
            
            # Guardo el historial
            self.history['train_loss'].append(train_loss.item())
            self.history['val_loss'].append(val_loss.item())
            self.history['val_accuracy'].append(val_accuracy)
            
            # Ajusto el learning rate
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardo el mejor modelo
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Imprimo el progreso
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Época {epoch+1:03d}/{self.config.epochs} | "
                      f"Loss Entreno: {train_loss.item():.4f} | "
                      f"Loss Val: {val_loss.item():.4f} | "
                      f"Precisión Val: {val_accuracy*100:.2f}%")
            
            # Early stopping si no mejora
            if patience_counter >= max_patience:
                print(f"\n⚠ Early stopping activado en época {epoch+1}")
                break
        
        # Cargo el mejor modelo
        self.model.load_state_dict(self.best_model_state)
        
        print()
        print("=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"🏆 Mejor precisión en validación: {max(self.history['val_accuracy'])*100:.2f}%")
        print()
    
    def evaluar(self, X_val, y_val):
        """
        Evalúo el modelo con métricas profesionales
        """
        print("=" * 60)
        print("📊 EVALUACIÓN DEL MODELO")
        print("=" * 60)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val).argmax(dim=1).cpu().numpy()
            y_true = y_val.cpu().numpy()
        
        # Calculo métricas
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')
        
        print(f"\n🎯 Precisión (Accuracy): {accuracy*100:.2f}%")
        print(f"📈 F1-Score: {f1:.4f}\n")
        
        # Reporte de clasificación detallado
        print("📋 REPORTE DETALLADO POR CLASE:")
        print("-" * 60)
        report = classification_report(
            y_true, predictions,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, predictions)
        self._plot_confusion_matrix(cm, self.label_encoder.classes_)
        
        return accuracy, f1, cm
    
    def _plot_confusion_matrix(self, cm, class_names):
        """
        Creo y guardo la matriz de confusión
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.tight_layout()
        
        # Guardo la figura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.results_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {path}")
        plt.close()
    
    def plot_historial(self):
        """
        Creo gráficas del historial de entrenamiento
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfica de pérdida
        axes[0].plot(self.history['train_loss'], label='Entrenamiento', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Validación', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Pérdida (Loss)', fontsize=12)
        axes[0].set_title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfica de precisión
        axes[1].plot(self.history['val_accuracy'], color='green', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Precisión', fontsize=12)
        axes[1].set_title('Precisión en Validación', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardo la figura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.results_dir, f'training_history_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Historial de entrenamiento guardado en: {path}")
        plt.close()
    
    def guardar_modelo(self, nombre="modelo"):
        """
        Guardo el modelo entrenado y todos sus componentes
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre}_{timestamp}.pkl"
        ruta = os.path.join(self.config.models_dir, nombre_archivo)
        
        # Guardo todo lo necesario para hacer predicciones después
        modelo_completo = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'input_size': self.model.network[0].in_features,
                'hidden_layers': self.config.hidden_layers,
                'output_size': self.model.network[-1].out_features,
                'dropout_rate': self.config.dropout_rate
            },
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'history': self.history,
            'config': self.config
        }
        
        joblib.dump(modelo_completo, ruta)
        print(f"\n💾 Modelo guardado exitosamente en: {ruta}")
        return ruta
    
    def cargar_modelo(self, ruta):
        """
        Cargo un modelo previamente guardado
        """
        print(f"📂 Cargando modelo desde: {ruta}")
        
        modelo_completo = joblib.load(ruta)
        
        # Recreo el modelo con la arquitectura guardada
        arch = modelo_completo['model_architecture']
        self.model = FlexibleMLP(
            input_size=arch['input_size'],
            hidden_layers=arch['hidden_layers'],
            output_size=arch['output_size'],
            dropout_rate=arch['dropout_rate']
        ).to(self.device)
        
        # Cargo los pesos
        self.model.load_state_dict(modelo_completo['model_state_dict'])
        self.model.eval()
        
        # Cargo el scaler y encoder
        self.scaler = modelo_completo['scaler']
        self.label_encoder = modelo_completo['label_encoder']
        self.history = modelo_completo['history']
        
        print("✓ Modelo cargado exitosamente")
        print(f"✓ Clases: {self.label_encoder.classes_}")
        print()
    
    def predecir(self, X_nuevo):
        """
        Hago predicciones sobre nuevos datos
        
        Args:
            X_nuevo: Array o DataFrame con las características
        
        Returns:
            predictions: Etiquetas predichas
            probabilities: Probabilidades de cada clase
        """
        # Convierto a numpy si es necesario
        if isinstance(X_nuevo, pd.DataFrame):
            X_nuevo = X_nuevo.values
        
        # Normalizo los datos
        X_nuevo = self.scaler.transform(X_nuevo)
        
        # Convierto a tensor
        X_tensor = torch.FloatTensor(X_nuevo).to(self.device)
        
        # Hago la predicción
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        # Decodifico las etiquetas
        predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities


# ========================================
# FUNCIÓN PRINCIPAL DE EJEMPLO
# ========================================

def ejemplo_uso_datos_sinteticos():
    """
    Ejemplo de uso con datos sintéticos (para demostración)
    """
    from sklearn.datasets import make_classification
    
    print("\n" + "=" * 60)
    print("🎓 DEMO: ENTRENAMIENTO CON DATOS SINTÉTICOS")
    print("=" * 60)
    print()
    
    # Creo datos sintéticos
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    # Inicializo el clasificador
    clasificador = ClasificadorProfesional()
    
    # Preparo los datos
    X_train, X_val, y_train, y_val = clasificador.preparar_datos(X, y)
    
    # Creo el modelo
    clasificador.crear_modelo(
        input_size=X.shape[1],
        output_size=len(np.unique(y))
    )
    
    # Entreno
    clasificador.entrenar(X_train, y_train, X_val, y_val)
    
    # Evalúo
    clasificador.evaluar(X_val, y_val)
    
    # Grafico el historial
    clasificador.plot_historial()
    
    # Guardo el modelo
    ruta_modelo = clasificador.guardar_modelo("modelo_ejemplo")
    
    # Ejemplo de predicción
    print("\n" + "=" * 60)
    print("🔮 EJEMPLO DE PREDICCIÓN")
    print("=" * 60)
    
    # Hago predicciones en algunos ejemplos
    X_prueba = X[:5]
    predicciones, probabilidades = clasificador.predecir(X_prueba)
    
    print("\nPredicciones en 5 ejemplos:")
    for i, (pred, probs) in enumerate(zip(predicciones, probabilidades)):
        print(f"Ejemplo {i+1}: Predicción = {pred} (probabilidad: {probs.max()*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETADA")
    print("=" * 60)
    print()


def ejemplo_uso_csv():
    """
    Ejemplo de cómo usar el clasificador con un archivo CSV real
    """
    print("\n" + "=" * 60)
    print("📝 INSTRUCCIONES PARA USAR CON TUS PROPIOS DATOS CSV")
    print("=" * 60)
    print("""
    Para usar este clasificador con tus propios datos:
    
    1. Prepara tu archivo CSV con:
       - Una columna con la variable objetivo (lo que quieres predecir)
       - Las demás columnas como características
    
    2. Usa el siguiente código:
    
    ```python
    # Inicializo el clasificador
    clasificador = ClasificadorProfesional()
    
    # Cargo mis datos desde CSV
    X, y = clasificador.cargar_datos_csv(
        ruta_csv="mis_datos.csv",
        columna_objetivo="columna_a_predecir",
        columnas_excluir=["id", "nombre"]  # Opcional
    )
    
    # Preparo los datos
    X_train, X_val, y_train, y_val = clasificador.preparar_datos(X, y)
    
    # Creo el modelo
    clasificador.crear_modelo(
        input_size=X.shape[1],
        output_size=len(np.unique(y))
    )
    
    # Entreno
    clasificador.entrenar(X_train, y_train, X_val, y_val)
    
    # Evalúo
    clasificador.evaluar(X_val, y_val)
    
    # Grafico resultados
    clasificador.plot_historial()
    
    # Guardo el modelo
    clasificador.guardar_modelo("mi_modelo")
    ```
    
    3. Para hacer predicciones más tarde:
    
    ```python
    # Cargo el modelo guardado
    clasificador = ClasificadorProfesional()
    clasificador.cargar_modelo("modelos_guardados/mi_modelo_20231113_120000.pkl")
    
    # Hago predicciones
    predicciones, probabilidades = clasificador.predecir(X_nuevo)
    ```
    """)


# ========================================
# PUNTO DE ENTRADA
# ========================================

if __name__ == "__main__":
    # Ejecuto la demostración
    ejemplo_uso_datos_sinteticos()
    
    # Muestro las instrucciones
    ejemplo_uso_csv()

