import torch
import numpy as np

def forecast_runner(test_loader,model,device,window_dec,dec_in_min,dec_in_max,n_batch_considered):
  predictions=[]
  test_labels=[]
  for i, batch in enumerate(test_loader): 
    _,_,lab=batch
    forecasted_label, _ = forecast(model, device, batch, window_dec, (dec_in_min, dec_in_max) )
    predictions.append(forecasted_label)
    test_labels.append(lab)
  pred=torch.stack(predictions)
  pred=torch.reshape(pred,(16*n_batch_considered,5,1,7,7))
  labels=torch.stack(test_labels)
  labels=torch.reshape(labels,(16*n_batch_considered,5,1,7,7))
  forecasted_label_p= np.transpose( pred.detach().numpy(),axes=(3,4,0,1,2))
  test_label_p= np.transpose( labels,axes=(3,4,0,1,2))
  return forecasted_label_p, test_label_p, pred, labels


def forecast(model, device, data, n_steps, dec_normalizer):
    
    enc_in, dec_in, lab = data
    
    assert enc_in.shape[0] == dec_in.shape[0]
    assert enc_in.shape[0] == lab.shape[0]
    n_data = enc_in.shape[0]
    
    enc_in=enc_in.to(device)
    dec_in = dec_in.to(device)
    
    dec_in_min, dec_in_max = dec_normalizer
    dec_in_min, dec_in_max = map (lambda x : torch.from_numpy(x).float().to(device), [dec_in_min, dec_in_max]) 
    dec_in_min, dec_in_max = map (lambda x : x.reshape( shape=(1,)+ x.shape ), [dec_in_min, dec_in_max]) 
    dec_in_min, dec_in_max = map (lambda x : torch.stack([x]*n_data), [dec_in_min, dec_in_max])
    
    forecasted = model.forecast([enc_in, dec_in], n_steps, dec_in_min, dec_in_max)
    
    forecast_info = {} #here add meta-info about forecast performance
    
    return forecasted, forecast_info