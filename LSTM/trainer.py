import random
import numpy as np
import torch
from tqdm import tqdm
from time import sleep
import random
def train_runner(model,n_epochs,opt,schedule,n_batch_validation,n_batch_train,train_loader,loss_fn,early_stopping,best_model_path,device):  
  lr=[]
  train_history=[]
  val_history=[]
  best_val_loss=10000
  for epoch in range(n_epochs):
          print(f"learnign rate = {opt.param_groups[0]['lr']}")
          lr.append(opt.param_groups[0]['lr'])
          indexes_of_val_batches = random.sample(range(len(train_loader)),k=n_batch_validation)
          
          val_batches=[]
          train_losses_per_batch=[]
          
          #training
          with tqdm(total=n_batch_train, ncols=100, desc=f'epoch {epoch+1}/{n_epochs}') as bar:
              model.train() 
              for i, batch in enumerate(train_loader):
                
                  if i not in indexes_of_val_batches:
                      
                      enc_in, dec_in, lab = batch
                      enc_in=enc_in.to(device)
                      dec_in = dec_in.to(device)
                      lab=lab.to(device)
                      pred = model([enc_in, dec_in])
                      loss = loss_fn(lab, pred)
                      
                      train_losses_per_batch.append(loss.item())
                      mean_loss = np.array(train_losses_per_batch).mean()
                      
                      loss.backward()
                      schedule.step_and_update_lr()
                      schedule.zero_grad()
                      
                      bar.update(1) 
                      bar.set_postfix({'train_loss' : np.round(mean_loss,3)})
                      
                  else:
                      val_batches.append(batch)
              model.eval()
              #validation 
              validation_losses_per_batch=[]
              for val_batch in val_batches:
                  
                  enc_in, dec_in, lab = val_batch
                  enc_in=enc_in.to(device)
                  dec_in = dec_in.to(device)
                  lab=lab.to(device)
                  pred = model([enc_in, dec_in])
                  val_loss = loss_fn(pred,lab)
                  validation_losses_per_batch.append(val_loss.item())
              
              mean_val_loss = np.array(validation_losses_per_batch).mean()
              early_stopping(mean_val_loss, model)
              if mean_val_loss < best_val_loss:
                  best_val_loss = mean_val_loss
                  torch.save(model.state_dict(), best_model_path)
              if early_stopping.early_stop:
                print("Early stopping")
                break
              bar.set_postfix({'train_loss':np.round(mean_loss,3), 'val_loss':np.round(mean_val_loss,3)}) 
              bar.close() 
          train_history.append(mean_loss.item())
          val_history.append(mean_val_loss.item())
  return train_history, val_history