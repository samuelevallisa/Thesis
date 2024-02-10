import torch 
import torch.nn as nn
from torch.nn import Conv2d


def conv_block(in_c, out_c, kernel_size):
  return nn.Sequential(
        Conv2d(in_c, out_c, kernel_size, padding='same'),
        Conv2d(out_c,out_c,kernel_size, padding='same'),
    )
class ConvLSTM(torch.nn.Module):
    def __init__(self, device, channels_in, hidden_dim, kernel_size, channels_out=None, pass_states = False, return_sequence = False):
        
        super(ConvLSTM, self).__init__()
        
        if return_sequence:
            if channels_out is None: raise TypeError("Please specify a number of channels for output sequence")
            self.out = Conv2d(hidden_dim, channels_out, kernel_size, padding='same')
        
        self.hidden_dim = hidden_dim
        self.forget = conv_block(channels_in+hidden_dim, hidden_dim,  kernel_size)#Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same')
        self.input = conv_block(channels_in+hidden_dim, hidden_dim,  kernel_size)#Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same')
        self.candidate = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #conv_block(channels_in+hidden_dim, hidden_dim,  kernel_size)#
        self.output = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #conv_block(channels_in+hidden_dim, hidden_dim,  kernel_size) #
        
        self.return_sequence = return_sequence
        self.pass_states = pass_states
        self.device = device
        
        
    def forward(self, x):
        
        
        if  not self.pass_states :
            hidden_state = torch.zeros((x.shape[0],)+(self.hidden_dim,)+x.shape[-2:]).to(self.device)
            cell_state = torch.zeros((x.shape[0],)+(self.hidden_dim,)+x.shape[-2:]).to(self.device)
        else: 
            x, hidden_state, cell_state = x

        outputs=[]
        for i in range(x.shape[1]):
            
            x_temp = x[:, i, :, :, :]
            
            x_conc = torch.cat([x_temp, hidden_state], dim=1)
            
            forg = torch.nn.Sigmoid()(self.forget(x_conc))
            inp= torch.nn.Sigmoid()(self.input(x_conc))
            cand = torch.nn.Tanh()(self.candidate(x_conc))
            out = torch.nn.Sigmoid()(self.output(x_conc))
            
            cell_state *= forg
            cell_state += (inp*cand)
            
            hidden_state = (torch.nn.Tanh()(cell_state))*out
            if self.return_sequence:
                #here we can change activation function for output seq!!
                outputs.append(torch.nn.ReLU()(self.out(hidden_state))) 
        if self.return_sequence: 
            return  hidden_state, cell_state, torch.stack(outputs, dim=1)
        else:
            return hidden_state, cell_state 


class EncDecConvLSTM(torch.nn.Module):
    
    def __init__(self, device, n_features, hidden_dim, n_outputs, kernel_size,training=True):
        
        super(EncDecConvLSTM, self).__init__()
        self.encoder = ConvLSTM(device, n_features, hidden_dim, kernel_size, pass_states = False, return_sequence = True)
        #self.decoder = ConvLSTM(device, n_outputs, hidden_dim, kernel_size, n_outputs, pass_states = True, return_sequence = True)
        self.h_state=None
        self.cell_state=None
        
    def forward(self, x,saved_h_and_c_state=False):
        enc_in, dec_in = x
        if not saved_h_and_c_state:
          h_enc,c_enc = self.encoder(enc_in)
          self.h_state=h_enc
          self.cell_state=c_enc

        _,_, output_seq = self.decoder([dec_in,self.h_state,self.cell_state])

        return output_seq