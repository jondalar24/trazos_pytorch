# utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import IMAGE_SIZE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#Función para visualizar los pesos de una capa convolucional
def plot_channels(W):
    n_out = W.shape[0]  # W tiene forma (n_out, n_in, k, k)
    n_in = W.shape[1]
    w_min = W.min().item() # Obtenemos el valor dentro del kernel más bajo convertido a numpy
    w_max = W.max().item()
    fig, axes = plt.subplots(n_out, n_in) # crea una cuadrícula filas x columnas
    fig.subplots_adjust(hspace=0.1) # espacio vertical ajustado
    
    #iteración para visualizar todos los kernels
    out_index = 0 # filtro que vamos a visualizar
    in_index = 0 # canal de entrada para el filtro
    # flat permite recorrer el array subplot de forma lineal
    for ax in axes.flat:
        # Si hemos visualizado todos los canales de entrada para un filtro,
        # pasamos al siguiente filtro y reiniciamos el contador de canal
        if in_index > n_in-1:
            out_index = out_index + 1
            in_index = 0
        # W[filtro, canal, (todo el array)]; visualizamos toda la matriz 2D
        # cmap='seismic' representa valores negativos en azul, positivos en rojo
        ax.imshow(W[out_index, in_index, :, :],vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_yticks([])
        ax.set_xticks([])
        in_index = in_index + 1
    
    plt.show()

# Función para visualizar los parámetros de una capa convolucional para un canal
def plot_parameters(W, number_rows=1, name="", i=0):
    # Extraemos solo los filtros del canal 'i'-ésimo de entrada
    # W tiene forma (n_out, n_in, k, k) → tomamos todos los filtros y solo un canal
    W = W.data[:, i, :, :]  # Nos quedamos con el canal i de cada filtro

    n_filters = W.shape[0]  # Número de filtros (n_out)
    
    # Obtenemos el valor mínimo y máximo para normalizar la escala de color
    w_min = W.min().item()
    w_max = W.max().item()

    # Creamos una cuadrícula de subplots para visualizar cada filtro
    # El número de columnas se calcula como n_filtros / filas
    fig, axes = plt.subplots(number_rows, n_filters // number_rows)

    # Ajuste del espacio vertical entre subplots
    fig.subplots_adjust(hspace=0.4)

    # Recorremos cada subplot (ax) para visualizar un filtro
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Etiquetamos cada gráfico con el número de filtro (1-indexed)
            ax.set_xlabel("kernel:{0}".format(i+1))
            # Mostramos los pesos del filtro i usando mapa de color 'seismic'
            # Rojo para valores positivos, azul para negativos, blanco para cero
            ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])

    # Título general del conjunto de filtros
    plt.suptitle(name, fontsize=10)
    plt.show()

# Función para visualizar las activaciones de una capa convolucional
def plot_activations(A, number_rows=1, name="", i=0):
    # Extraemos solo la primera imagen del batch y convertimos a NumPy
    # A tiene forma (batch_size, canales, alto, ancho). Tomamos solo el primer elemento.
    A = A[0, :, :, :].detach().numpy()  

    # Número de mapas de activación generados (uno por filtro aplicado)
    n_activations = A.shape[0]

    # Obtenemos el valor mínimo y máximo de todas las activaciones para normalizar la escala de color
    A_min = A.min().item()
    A_max = A.max().item()

    # Creamos una cuadrícula de subgráficas: cada celda mostrará un mapa de activación
    # El número de columnas se calcula dividiendo el total de activaciones entre las filas
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    
    # Ajustamos el espacio vertical entre subgráficas para que no se solapen
    fig.subplots_adjust(hspace=0.4)

    # Recorremos cada subplot de forma lineal
    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Añadimos etiqueta con el número del mapa de activación
            ax.set_xlabel(f"activation: {i + 1}")

            # Mostramos el mapa de activación i con escala de color centrada (rojo negativo, azul positivo)
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')

            # Quitamos los ticks de los ejes para limpiar la visualización
            ax.set_xticks([])
            ax.set_yticks([])

    # Añadimos un título general si se especificó
    if name:
        plt.suptitle(name, fontsize=12)

    # Mostramos la figura final con todas las activaciones
    plt.show()

# función auxiliar para visualizar una imagen y su etiqueta asociada. 
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))

def visualize_predictions(model, data_loader, device, n_images=10):
    model.eval()
    model.to(device)

    # Obtenemos un único batch del data_loader
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    images = images.to(device)
    labels = labels.to(device)

    # Realizamos predicción
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Mostramos n imágenes
    plt.figure(figsize=(15, 4))
    for idx in range(n_images):
        image = images[idx].cpu().squeeze().numpy()
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        correct = (true_label == pred_label)

        plt.subplot(1, n_images, idx + 1)
        plt.imshow(image, cmap='gray')
        color = 'green' if correct else 'red'
        plt.title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        plt.axis('off')

    plt.suptitle("Visualización de predicciones del modelo", fontsize=14)
    plt.show()

def plot_wrong_predictions(model, data_loader, device='cpu', max_images=10):
    """
    Muestra imágenes donde el modelo ha fallado: predicción ≠ etiqueta real.
    """
    model.eval()
    wrong_images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            z = model(inputs)
            _, preds = torch.max(z, 1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    wrong_images.append(inputs[i].cpu())
                    true_labels.append(labels[i].item())
                    pred_labels.append(preds[i].item())

                if len(wrong_images) >= max_images:
                    break
            if len(wrong_images) >= max_images:
                break

    # Plot
    fig, axes = plt.subplots(1, len(wrong_images), figsize=(15, 4))
    for idx, ax in enumerate(axes):
        image = wrong_images[idx].squeeze()
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {true_labels[idx]}\nPred: {pred_labels[idx]}')
        ax.axis('off')
    plt.suptitle("Ejemplos de Predicciones Incorrectas", fontsize=14)
    plt.show()


def plot_confusion_matrix(model, data_loader, device='cpu'):
    """
    Evalúa el modelo en un conjunto de datos y muestra la matriz de confusión.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()