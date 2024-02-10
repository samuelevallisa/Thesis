import torch 
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Linear, Module, ReLU, Tanh, Sigmoid


class HiddenBlock(nn.Module):
    
    def __init__(self, channels_in, hidden_dim, kernel_size):
        
        super(HiddenBlock, self).__init__()
        
        self.conv1=Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same')
        #self.batchnorm1=BatchNorm2d(hidden_dim)
        self.conv2=Conv2d(hidden_dim, hidden_dim,  kernel_size, padding='same')
        #self.batchnorm2=BatchNorm2d(hidden_dim)
        self.conv3=Conv2d(hidden_dim, hidden_dim,  kernel_size, padding='same')
        #self.conv4=Conv2d(hidden_dim, hidden_dim,  kernel_size, padding='same')
        #self.conv5=Conv2d(hidden_dim, hidden_dim,  kernel_size, padding='same')
        
    def forward(self, x):
        
        x = self.conv1(x)
        #x = ReLU()
        x = self.conv2(x)
        #x = ReLU()
        x=self.conv3(x)
        #x=self.conv4(x)
        #x=self.conv5(x)
        
        
        return x


    
class OutputBlock(Module):
    pass

def conv_block(in_c, out_c, kernel_size):
  return nn.Sequential(
        Conv2d(in_c, out_c, kernel_size, padding='same'),
        Conv2d(out_c,out_c,kernel_size, padding='same'),
    )
        
class ENC_ConvLSTM(Module):
    
    def __init__(self, device, channels_in, hidden_dim, kernel_size, channels_out=None, pass_states = False, return_sequence = False):
        
        super(ENC_ConvLSTM, self).__init__()
        
        if return_sequence:
            if channels_out is None: raise TypeError("Please specify a number of channels for output sequence")
            
            #here we could use "OutputBlock"
            self.out = Conv2d(hidden_dim, channels_out + hidden_dim , kernel_size, padding='same') # HiddenBlock(hidden_dim, channels_out+hidden_dim,  kernel_size) #
            self.lin_1 = Linear(channels_out+hidden_dim,channels_out)
            #self.lin_2 = Linear(16,channels_out)
            #self.Relu = ReLU()
        
        self.hidden_dim = hidden_dim
        
        #here we could use "HiddenBlock"
        self.forget = Linear(channels_in+hidden_dim,hidden_dim)#Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same') 
        self.input = Linear(channels_in+hidden_dim,hidden_dim)#Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #HiddenBlock(channels_in, hidden_dim,  kernel_size)#
        self.candidate = Linear(channels_in+hidden_dim,hidden_dim)#Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size) #
        self.output =Linear(channels_in+hidden_dim,hidden_dim)#Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size)
        
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
            
            forg = Sigmoid()(self.forget(torch.transpose(x_conc,1,3)))
            inp= Sigmoid()(self.input(torch.transpose(x_conc,1,3)))
            cand = Tanh()(self.candidate(torch.transpose(x_conc,1,3)))
            out = Sigmoid()(self.output(torch.transpose(x_conc,1,3)))
            
            forg=torch.transpose(forg,1,3)
            inp=torch.transpose(inp,1,3)
            cand=torch.transpose(cand,1,3)
            cell_state *= forg
            cell_state += (inp*cand)
            out=torch.transpose(out,1,3)
            hidden_state = (Tanh()(cell_state))*out
            if self.return_sequence:
              #here we can change activation function for output seq!! (if we use "OutputBlock" we add ReLU there)
              rho=self.out(hidden_state)
              rho=torch.transpose(rho,1,3)
              rho=self.lin_1(rho)
              #rho=self.Relu(rho)
              outputs.append(torch.transpose(rho,1,3))

        if self.return_sequence: 
            return  hidden_state, cell_state, torch.stack(outputs, dim=1)
        else:
            return hidden_state, cell_state 

class DEC_ConvLSTM(Module):
    
    def __init__(self, device, channels_in, hidden_dim, kernel_size, channels_out=None, pass_states = False, return_sequence = False):
        
        super(DEC_ConvLSTM, self).__init__()
        
        if return_sequence:
            if channels_out is None: raise TypeError("Please specify a number of channels for output sequence")
            
            #here we could use "OutputBlock"
            self.out = Conv2d(hidden_dim, channels_out + hidden_dim , kernel_size, padding='same') # HiddenBlock(hidden_dim, channels_out+hidden_dim,  kernel_size) #
            self.lin_1 = Linear(channels_out+hidden_dim,channels_out)
            #self.lin_2 = Linear(16,channels_out)
            #self.Relu = ReLU()
        
        self.hidden_dim = hidden_dim
        
        #here we could use "HiddenBlock"
        self.forget = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same') 
        self.input = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #HiddenBlock(channels_in, hidden_dim,  kernel_size)#
        self.candidate = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size) #
        self.output = Linear(channels_in+hidden_dim,hidden_dim)#Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size)
        
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
            
            forg = Sigmoid()(self.forget(x_conc))
            inp= Sigmoid()(self.input(x_conc))
            cand = Tanh()(self.candidate(x_conc))
            out = Sigmoid()(self.output(torch.transpose(x_conc,1,3)))
            
            #forg=torch.transpose(forg,1,3)
            #inp=torch.transpose(inp,1,3)
            #cand=torch.transpose(cand,1,3)
            cell_state *= forg
            cell_state += (inp*cand)
            out=torch.transpose(out,1,3)
            hidden_state = (Tanh()(cell_state))*out
            if self.return_sequence:
              #here we can change activation function for output seq!! (if we use "OutputBlock" we add ReLU there)
              rho=self.out(hidden_state)
              rho=torch.transpose(rho,1,3)
              rho=self.lin_1(rho)
              #rho=self.Relu(rho)
              outputs.append(torch.transpose(rho,1,3))

        if self.return_sequence: 
            return  hidden_state, cell_state, torch.stack(outputs, dim=1)
        else:
            return hidden_state, cell_state

class EncDecConvLSTM(Module):
    
    def __init__(self, device, n_features, hidden_dim, n_outputs, kernel_size ):
        
        super(EncDecConvLSTM, self).__init__()
        self.encoder = ENC_ConvLSTM(device, n_features, hidden_dim, kernel_size, pass_states = False, return_sequence = False)
        self.decoder = DEC_ConvLSTM(device, n_outputs, hidden_dim, kernel_size, n_outputs, pass_states = True, return_sequence = True)
        
        
    def forward(self, x):
        enc_in, dec_in = x
        h_enc,c_enc = self.encoder(enc_in)
        _,_, output_seq = self.decoder([dec_in,h_enc,c_enc])

        return output_seq
    
    def forecast(self, x, n_steps, dec_in_min, dec_in_max):
        
        enc_in, dec_in = x #here dec_in is a 1-timestep token of 0s
        assert dec_in.shape[1]==1, "in forecast, decoder input must be a one-timestep token"

        h_to_dec,c_to_dec = self.encoder(enc_in)
        forecasted_seq=[]
        
        for i in range(n_steps):
            
            h_to_dec, c_to_dec, dec_out = self.decoder([dec_in,h_to_dec,c_to_dec])
            forecasted_seq.append(dec_out)
            dec_in = (dec_out-dec_in_min) / (dec_in_max-dec_in_min)
        
        return torch.cat(forecasted_seq, dim=1)