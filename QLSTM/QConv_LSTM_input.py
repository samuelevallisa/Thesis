import torch 
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Linear, Module, ReLU, Tanh, Sigmoid
from torch.nn.functional import unfold, pad
import pennylane as qml

class QConv2d(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, wires=6, stride=1, padding='same'):
        super(QConv2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 
        self.wires = wires 
        
        self.device = qml.device("default.qubit", wires=self.wires)
        
        self.define_spec() #define 
        self.define_circuit()
        
        self.qlayer = qml.qnn.TorchLayer( self.qnode, self.weight_shapes )
        
        
    def define_spec(self):
        
        if isinstance(self.kernel_size, tuple) or isinstance(self.kernel_size, list):
            self.k_height, self.k_width = self.kernel_size
            kernel_entries = self.k_height * self.k_width * self.in_channels
        elif isinstance(self.kernel_size, int):
            self.k_height = self.k_width = self.kernel_size
            kernel_entries = self.k_height * self.k_width * self.in_channels
            
        else:
            raise ValueError('kernel_size must be either an int or a tuple containing kernel dimensions (kernel_height, kernel_width)')
        
        if isinstance(self.stride, tuple) or isinstance(self.stride, list):
            self.s_height, self.s_width = self.stride
        elif isinstance(self.stride, int):
            self.s_height = self.s_width = self.stride
        else:
            raise ValueError('stride must be either an int or a tuple containing stride values over two axis (stride_height, stride_width)')
        
        if self.padding == 'same': #FIXME: 'same' padding not properly computed 
            
            p_height = self.k_height - 1
            p_width = self.k_width - 1
            if p_height % 2 == 0: 
                self.p_top = self.p_bottom = int(p_height/2)
            else:
                self.p_top, self.p_bottom = int(p_height/2) + 1, int(p_height/2)   
            if p_width % 2 == 0: 
                self.p_left = self.p_right = int(p_width/2)
            else:
                self.p_left, self.p_right = int(p_width/2) + 1, int(p_width/2)
            
            #@NotImplemented
            #to compute padding that always maintain same dimension: (need height and width of input!)
            #p_height = self.k_heigth + self.height *(self.s_heigth - 1) - self.s_heigth
            #p_width = self.k_width + self.width *(self.s_width - 1) - self.s_width
        
        elif isinstance(self.padding, tuple) or isinstance(self.padding, list):
            self.p_left, self.p_right, self.p_top, self.p_bottom  = self.padding
        
        elif self.padding == 'valid':
            pass
            
        else:
            raise ValueError("padding must be either 'same' or 'valid' or a four element tuple indicating (left pad, right pad, top pad, bottom pad)")
        
        #self.wires = math.ceil( math.log(kernel_entries, 2) )
        
    def define_circuit(self):
        
        @qml.qnode(self.device)
        def qnode(inputs, weights_0,weights_1,weights_2,weights_3,weights_4,weights_5,weights_6,weights_7,weights_8,weights_9,weights_10,weights_11):
            
            qml.AmplitudeEmbedding( inputs, wires = range(self.wires), normalize=True, pad_with=0 )
            
            qml.RX(weights_0,wires=0)
            qml.RY(weights_1,wires=0)
            qml.RX(weights_2,wires=1)
            qml.RY(weights_3,wires=1)
            qml.RX(weights_4,wires=2)
            qml.RY(weights_5,wires=2)
            qml.RX(weights_6,wires=3)
            qml.RY(weights_7,wires=3)
            qml.RX(weights_8,wires=4)
            qml.RY(weights_9,wires=4)
            qml.RX(weights_10,wires=5)
            qml.RY(weights_11,wires=5)
    
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[5, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]#qml.probs(wires =  list(range(self.wires))) #qml.e[xpval(qml.PauliZ(i)) for i in range(self.wires)]#
        
        #self.weight_shapes = {"weights_0":(3,6)}
        self.weight_shapes = { "weights_0": 1, "weights_1": 1, "weights_2": 1,
                             "weights_3": 1, "weights_4": 1, "weights_5": 1,
                             "weights_6": 1, "weights_7": 1, "weights_8": 1,
                             "weights_9": 1, "weights_10": 1, "weights_11": 1
                             } 
        self.qnode = qnode
   
    def forward(self, x):
    
        height, width = x.shape[-2:]
        
        if self.padding != 'valid':
            x = pad( x, (self.p_left, self.p_right, self.p_top, self.p_bottom ) )
        
        out_shape = ( int( ( height + self.p_top + self.p_bottom - self.k_height ) / self.s_height ) + 1 , 
                      int( ( width + self.p_left + self.p_right - self.k_width ) / self.s_width ) + 1 )
        x = torch.transpose( unfold( x, kernel_size=self.kernel_size, stride=self.stride) , -1, -2 )
        x = torch.transpose(self.qlayer(x) , -1, -2 )
        
        return torch.reshape(x, x.shape[:2] + out_shape)
    


def conv_block(in_c, out_c, kernel_size):
  return nn.Sequential(
        Conv2d(in_c, out_c, kernel_size, padding='same'),
        Conv2d(out_c,out_c,kernel_size, padding='same'),
    )
        
class ConvLSTM(Module):
    
    def __init__(self, device, channels_in, hidden_dim, kernel_size, channels_out=None, pass_states = False, return_sequence = False):
        
        super(ConvLSTM, self).__init__()
        
        if return_sequence:
            if channels_out is None: raise TypeError("Please specify a number of channels for output sequence")
            
            #here we could use "OutputBlock"
            self.out = Conv2d(hidden_dim, channels_out + hidden_dim , kernel_size, padding='same') # HiddenBlock(hidden_dim, channels_out+hidden_dim,  kernel_size) #
            self.lin_1 = Linear(channels_out+hidden_dim,channels_out)
            #self.lin_2 = Linear(16,channels_out)
            #self.Relu = ReLU()
        
        self.hidden_dim = hidden_dim
        self.qubits_passed = 4
        #here we could use "HiddenBlock"
        self.forget = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same') 
        self.input = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #HiddenBlock(channels_in, hidden_dim,  kernel_size)#
        self.candidate = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size) #
        self.output = Linear(channels_in+hidden_dim,hidden_dim) #Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size)
        
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


class QuantumConvLSTM(Module):
    
    def __init__(self, device, channels_in, hidden_dim, kernel_size, channels_out=None, pass_states = False, return_sequence = False):
        
        super(QuantumConvLSTM, self).__init__()
        
        if return_sequence:
            if channels_out is None: raise TypeError("Please specify a number of channels for output sequence")
            
            #here we could use "OutputBlock"
            self.out = Conv2d(hidden_dim, channels_out + hidden_dim , kernel_size, padding='same') # HiddenBlock(hidden_dim, channels_out+hidden_dim,  kernel_size) #
            self.lin_1 = Linear(channels_out+hidden_dim,channels_out)
            #self.lin_2 = Linear(16,channels_out)
            #self.Relu = ReLU()
        
        self.hidden_dim = hidden_dim
        self.qubits_passed = 6
        #here we could use "HiddenBlock"
        self.forget = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') #Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same') 
        self.input = QConv2d(channels_in+hidden_dim,  kernel_size, wires=self.qubits_passed) #HiddenBlock(channels_in, hidden_dim,  kernel_size)#
        self.candidate = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size) #
        self.output = Linear(channels_in+hidden_dim,hidden_dim) #Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same') # HiddenBlock(channels_in, hidden_dim,  kernel_size)
        
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
        self.encoder = ConvLSTM(device, n_features, hidden_dim, kernel_size, pass_states = False, return_sequence = False)
        self.decoder = QuantumConvLSTM(device, n_outputs, hidden_dim, kernel_size, n_outputs, pass_states = True, return_sequence = True)
        
        
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