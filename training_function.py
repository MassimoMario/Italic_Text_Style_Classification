import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_function(pred_labels, labels, loss_fn = nn.CrossEntropyLoss()):
    L = loss_fn(pred_labels.view(pred_labels.shape[1], pred_labels.shape[2]), labels.view(-1))

    return L


# -------------------------------------------------------------------------------------------------- #



def training(model, train_loader, val_loader, num_epochs, lr = 4e-4, title = 'Training'):
    params = list(model.parameters())

    # three different optimizer
    optimizer = torch.optim.Adam(params, lr = lr)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        average_loss = 0.0
        val_loss = 0.0
        average_val_loss = 0.0

        for  i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            

            # forward pass through VAE
            pred_labels = model(data)
            
            # comuting total training loss
            loss_tot = loss_function(pred_labels.to(device),
                                     labels.to(device))
            
            loss_tot.backward()
            train_loss += loss_tot.item()


            optimizer.step()
            
            if (i + 1) % 5000 == 0:
                print(f'Train Epoch: {epoch+1} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss_tot.item() / len(data):.6f}')
        
        

        with torch.no_grad():
            for i, (data, labels) in enumerate(val_loader):
                data = data.to(device)
                labels = labels.to(device)

                labels = labels.type(torch.LongTensor)

                # forward pass through VAE
                pred_labels = model(data)
                
                
                # comuting total validation loss
                val_loss_tot = loss_function(pred_labels.to(device),
                                     labels.to(device))
                
                val_loss += val_loss_tot.item()


                
                if (i + 1) % 5000 == 0:
                    print(f'Train Epoch: {epoch+1} [{i * len(data)}/{len(val_loader.dataset)} ({100. * i / len(val_loader):.0f}%)]\tLoss: {val_loss_tot.item() / len(data):.6f}')
            
            
        average_loss = train_loss / len(train_loader.dataset)
        train_losses.append(average_loss)

        average_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)
        
        # printing average training and validation losses
        print(f'====> Epoch: {epoch+1} Average train loss: {average_loss:.4f}, Average val loss: {average_val_loss:.4f}')
    
    # plotting training and validation curve at the end of the for loop 
    plt.plot(np.linspace(1,num_epochs,len(train_losses)), train_losses, c = 'darkcyan',label = 'train')
    plt.plot(np.linspace(1,num_epochs,len(val_losses)), val_losses, c = 'orange',label = 'val')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.show()

    return train_losses