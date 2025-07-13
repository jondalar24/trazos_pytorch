# Trazos MNIST con PyTorch – Visualizando una Red Convolucional

Este proyecto muestra cómo construir y visualizar una red neuronal convolucional (CNN) desde cero usando **PyTorch**, aplicada al clásico dataset **MNIST** (dígitos escritos a mano).

**Objetivo**: No solo clasificar correctamente los dígitos, sino **entender cómo aprende la red** visualizando:

- Los **pesos (kernels)** antes y después del entrenamiento.
- Las **activaciones intermedias** en cada capa.
- Una **matriz de confusión** al finalizar.
- Las **predicciones correctas e incorrectas**.

##  Arquitectura del modelo

Modelo CNN compacto diseñado para entrenar sin GPU:

- Entrada: imágenes **16x16** en escala de grises (redimensionadas desde 28x28).
-  `Conv2D(1 → 16)` + `ReLU` + `MaxPool(2x2)`
-  `Conv2D(16 → 32)` + `ReLU` + `MaxPool(2x2)`
-  `Linear(512 → 10)` para clasificación final (10 dígitos)

 Todo implementado como clase `CNN(nn.Module)`, y con funciones auxiliares para visualizar pesos y activaciones.

##  Estructura del repositorio

```
trazos_pytorch/
├── cnn_mnist.py        # Definición de la red convolucional
├── dataset.py          # Preprocesado y carga de datos MNIST
├── utils.py            # Funciones de visualización y entrenamiento
├── main.py             # Script principal de ejecución
├── requirements.txt    # Dependencias necesarias
└── README.md           # Este archivo
```

##  Cómo ejecutar el proyecto

1. **Clona el repositorio:**

```bash
git clone https://github.com/jondalar24/trazos_pytorch.git
cd trazos_pytorch
```

2. **Instala las dependencias (recomendado en un entorno virtual):**

```bash
pip install -r requirements.txt
```

3. **Ejecuta el script principal:**

```bash
python main.py
```

 El modelo entrenará durante 5 épocas y mostrará:

- Curvas de pérdida y precisión
- Visualización de filtros antes y después del entrenamiento
- Activaciones intermedias
- Matriz de confusión
- Ejemplos de aciertos y errores

##  Ejemplo de visualización

![Activaciones ejemplo](ruta/a/tu/imagen.png)  
*Ejemplo de activaciones en la primera capa convolucional.*

##  Motivación

Aunque frameworks como Keras permiten prototipar muy rápido, PyTorch ofrece una visión más transparente y flexible del proceso de entrenamiento. Esta práctica está diseñada para **ver y entender lo que ocurre dentro de una CNN**, más allá de su rendimiento.

##  Recursos

- Dataset: [MNIST – Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- Framework: [PyTorch](https://pytorch.org/)
- Visualización: `matplotlib`

##  Autor

Ángel Calvar Pastoriza  
📍 [LinkedIn](https://www.linkedin.com/in/angelcalvar/)  
 Siempre abierto a feedback y colaboración.

---

## Hashtags

```
#DeepLearning #PyTorch #ComputerVision #MNIST #NeuralNetworks  
#MachineLearning #AI #IAExplicativa #AprendizajeProfundo #Python
```
