import numpy as np
from tqdm import tqdm
import random
import torch


def train(model,n_epochs,opt,schedule,n_batch_validation,n_batch_train,train_loader,loss_fun,early_stopping):
    lr=[]
    train_history=[]
    val_history=[]
    train_mape_history=[]
    val_mape_history=[]
    for epoch in range(n_epochs):
            print(f"learnign rate = {opt.param_groups[0]['lr']}")
            lr.append(opt.param_groups[0]['lr'])
            indexes_of_val_batches = random.sample(range(len(train_loader)),k=n_batch_validation)
            
            val_batches=[]
            train_losses_per_batch=[]
            mape_per_batch=[]
            validation_mape_per_batch=[]
            
            #training
            with tqdm(total=n_batch_train, ncols=100, desc=f'epoch {epoch+1}/{n_epochs}') as bar:
                model.train() 
                for i, batch in enumerate(train_loader):
                
                    if i not in indexes_of_val_batches:
                        
                        inputs, lab = batch
                        pred = model(inputs)
                        loss = loss_fun(pred,lab)
                        mape = (torch.abs(pred - lab) / lab).mean()
                        
                        train_losses_per_batch.append(loss.item())
                        mape_per_batch.append(mape.item())
                        mean_loss = np.array(train_losses_per_batch).mean()
                        mean_mape = np.array(mape_per_batch).mean()
                        
                        loss.backward()
                        schedule.step_and_update_lr()
                        schedule.zero_grad()
                        
                        bar.update(1) 
                        bar.set_postfix({'train_loss' : np.round(mean_loss,3), 'mape' : np.round(mean_mape,3)})
                        
                    else:
                        val_batches.append(batch)
                model.eval()
                #validation 
                validation_losses_per_batch=[]
                for val_batch in val_batches:
                    
                    inputs, lab = val_batch
                    pred = model(inputs)
                    val_loss = loss_fun(pred,lab)
                    val_mape = (torch.abs(pred - lab) / lab).mean()
                    validation_losses_per_batch.append(val_loss.item())
                    validation_mape_per_batch.append(val_mape.item())
                
                mean_val_loss = np.array(validation_losses_per_batch).mean()
                mean_val_mape = np.array(validation_mape_per_batch).mean()
                early_stopping(mean_val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                bar.set_postfix({'train_loss':np.round(mean_loss,3), 'train_mape' : np.round(mean_mape,3)})
                bar.close() 
                print(f"val_loss:{np.round(mean_val_loss,3)}") 
                print(f"val_mape:{np.round(mean_val_mape,3)}")
            train_history.append(mean_loss.item())
            val_history.append(mean_val_loss.item())
            train_mape_history.append(mean_mape.item())
            val_mape_history.append(mean_val_mape.item())
    return train_history, val_history, train_mape_history, val_mape_history