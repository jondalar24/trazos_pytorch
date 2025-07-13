import torch.utils
from torchvision import datasets, transforms
import torch

IMAGE_SIZE = 16

composed  = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Redim. a 16x16
    transforms.ToTensor() # Convertir a tensor (valores entre 0 y 1)
])

def get_dataloaders(batch_train=100, batch_val=5000):
    """
    Retorna los Dataloaders de entrenamiento y validación
    """
    # Descarga y carga del dataset MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=composed)
    validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=composed)

    # Carga en DataLoaders (batch de 100 para entrenamiento, 5000 para validación)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_val, shuffle=False)

    return train_loader, val_loader