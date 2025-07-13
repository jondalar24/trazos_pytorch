# entrenamiento
import matplotlib.pyplot as plt
import numpy as np
import torch

# Entrena el modelo y devuelve coste y precisión por época
def train_model(model, train_loader, validation_loader, criterion, optimizer, n_epochs, device):
    cost_list = []
    accuracy_list = []

    model.to(device)

    N_test = len(validation_loader.dataset)
    print("Entrenando....")

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()            
            z = model(x_batch)
            loss = criterion(z, y_batch)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()            

        cost_list.append(running_loss)            

        # Validación
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_val, y_val in validation_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                z = model(x_val)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_val).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)

        print(f"Época [{epoch+1}/{n_epochs}] - Pérdida: {running_loss:.4f} - Precisión: {accuracy*100:.2f}%")

    return cost_list, accuracy_list


# Gráfico coste y precisión por época
def plot_training_metrics(cost_list, accuracy_list):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(cost_list, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('Cost', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(accuracy_list, color=color)
    ax2.set_ylabel('Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()