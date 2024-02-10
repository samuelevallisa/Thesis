import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import pennylane as qml
from pennylane import numpy as np
import torch
from torch.nn.functional import unfold, pad

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
        def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5):
            
            qml.AmplitudeEmbedding( inputs, wires = range(self.wires), normalize=True, pad_with=0 )
            
            qml.Rot(*weights_0, wires=0)
            qml.Rot(*weights_1, wires=1)
            qml.Rot(*weights_2, wires=2)
            qml.Rot(*weights_3, wires=3)
            qml.Rot(*weights_4, wires=4)
            qml.Rot(*weights_5, wires=5)
            
            
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[5, 0])
            
            return qml.probs( wires = list(range(self.wires)) ) #[qml.expval(qml.PauliZ(i)) for i in range(self.wires)]
        
        self.weight_shapes = { "weights_0": 3, "weights_1": 3, "weights_2": 3,
                              "weights_3": 3, "weights_4": 3, "weights_5": 3}  
        self.qnode = qnode
   
    def forward(self, x):
    
        height, width = x.shape[-2:]
        
        if self.padding != 'valid':
            x = pad( x, (self.p_left, self.p_right, self.p_top, self.p_bottom ) )
        
        out_shape = ( int( ( height + self.p_top + self.p_bottom - self.k_height ) / self.s_height ) + 1 , 
                      int( ( width + self.p_left + self.p_right - self.k_width ) / self.s_width ) + 1 )
        #32,7,10,10
        print("before unfold shape")
        print(x.shape)
        x = torch.transpose( unfold( x, kernel_size=self.kernel_size, stride=self.stride) , -1, -2 )
        #32,64,63
        print("unfolded shape")
        print(x.shape)
        x=self.qlayer(x)
        print("convoluted shape")
        print(x.shape)
        x = torch.transpose( x , -1, -2 )
        print("transposed shape")
        print(x.shape)
        x=torch.reshape(x, x.shape[:2] + out_shape)
        print("final shape")
        print(x.shape)
        return x
        

def conv_block(in_c, kernel_size):
  return nn.Sequential(
        QConv2d(in_c, kernel_size=kernel_size),
        #nn.BatchNorm2d(hidden_dim_1),
        nn.ReLU(),
        #Conv2d(hidden_dim_1,hidden_dim_1*2,kernel_size, padding='same'),
        #Conv2d(hidden_dim_1*2, hidden_dim_1*4,kernel_size, padding='same'),
        #Conv2d(hidden_dim_1,out_c,kernel_size,padding='same'),
        #nn.ReLU()
    )

def dense_block(input_size, output_size, dropout_rate=0.1):
    return nn.Sequential(
        Linear(input_size,16),
        nn.ReLU(),
        #nn.Dropout(dropout_rate),
        Linear(16,8),
        nn.ReLU(),
        Linear(8,output_size)
    )

class RegressionModel(torch.nn.Module):
    def __init__(self, num_features_input, num_features_output , kernel_size,dx,dy):
        super(RegressionModel, self).__init__()


        self.convolutional_block=conv_block(num_features_input,kernel_size)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_block = dense_block(64*dx*dy,num_features_output)

    
    def forward(self,x):

        x=self.convolutional_block(x)
        x=self.flatten_layer(x)
        x=self.dense_block(x)

        return x

        


        


    