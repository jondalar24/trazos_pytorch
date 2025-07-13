import torch 
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, out_1=16, out_2=32):
        super(CNN, self).__init__()
        # Primera capa convolucional: 1 canal de entrada (escala de grises), out_1 filtros
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=out_1, kernel_size=5, padding=2
            )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Segunda capa convolucional: recibe out_1 canales, produce out_2 filtros
        self.cnn2 = nn.Conv2d(
            in_channels=out_1, out_channels=out_2, kernel_size=5,
            stride=1, padding=2
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Capa fully-connected (aplana 4x4xout_2 → 10 clases)
        self.fc1 = nn.Linear(out_2 * 4 * 4, 10)
    
    def forward(self, x):
        # Forward con activación ReLU + MaxPool tras cada capa convolucional
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    def activations(self, x):
        # Método auxiliar para obtener activaciones internas
        z1 = self.cnn1(x)
        a1 = torch.relu(z1)
        out = self.maxpool1(a1)
        
        z2 = self.cnn2(out)
        a2 = torch.relu(z2)
        out1 = self.maxpool2(a2)
        out = out.view(out.size(0),-1)
        return z1, a1, z2, a2, out1, out

