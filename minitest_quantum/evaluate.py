import numpy as np
from tqdm import tqdm
import torch

def evaluate(model, test_loader,loss_fun):
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store loss and accuracy values
    losses = []
    accuracies = []
    
    # Create a progress bar using tqdm
    pbar = tqdm(test_loader)
    
    # Loop over the test set
    for data, target in pbar:
        # Forward pass
        output = model(data)
        loss = loss_fun(output, target)
        
        # Compute MAPE as the accuracy metric
        mape = (torch.abs(output - target) / target).mean()
        
        
        # Append the loss and accuracy values to the lists
        losses.append(loss.item())
        accuracies.append(mape.item())
        mean_test_loss = np.array(losses).mean()
        mean_test_mape = np.array(accuracies).mean()
        # Update the progress bar
        pbar.set_description(f'Loss: {mean_test_loss.item():.4f} MAPE: {mean_test_mape.item():.4f}')
    
    return losses, accuracies