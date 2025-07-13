# main.py
import torch
import torch.nn as nn
from cnn_mnist import CNN
from dataset import get_dataloaders, IMAGE_SIZE
from utils import plot_parameters, visualize_predictions, plot_wrong_predictions
from train import train_model, plot_training_metrics
from utils import plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definición de modelo y parámetros
model = CNN(out_1=16, out_2=32)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 5

# Carga de los dataloaders
train_loader, val_loader = get_dataloaders()

# visualización de pesos iniciales
print("Pesos de la primera capa convolucional pre-entrenamiento")
plot_parameters(model.state_dict()['cnn1.weight'], number_rows=4, name='primera capa antes entreno' )

print("Pesos de la segunda capa convolucional pre-entrenamiento")
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=4, name='segunda capa antes entreno' )

# Entrenamos el modelo
cost_list, accuracy_list = train_model(
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    n_epochs=n_epochs,    
    device=device
)

# Visualización de métricas
print(" Visualizando evolución de coste y precisión")
plot_training_metrics(cost_list, accuracy_list)

# Visualizar pesos después del entrenamiento
print("Pesos tras entrenamiento")
plot_parameters(model.state_dict()['cnn1.weight'], number_rows=4, name="CNN1 - después del entrenamiento")
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=4, name="CNN2 - después del entrenamiento")
plot_confusion_matrix(model, val_loader, device=device)

visualize_predictions(
    model=model,
    data_loader=val_loader,
    device=device,
    n_images=10  # Número de imágenes a mostrar
)

plot_wrong_predictions(
    model=model,
    data_loader=val_loader,
    device=device,
    max_images=10
)


