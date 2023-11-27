import os.path

import torch
import torch.nn as nn
import time

class lstm_for_demonstration(nn.Module):
    def __init__(self,in_dim,out_dim,depth):
        super(lstm_for_demonstration,self).__init__()
        self.lstm=nn.LSTM(in_dim,out_dim,depth)

    def forward(self,inputs,hidden):
        out,hidden=self.lstm(inputs,hidden)
        return out,hidden


torch.manual_seed(29592)

model_dimension=8
sequence_length=20
batch_size=1
lstm_depth=1

inputs=torch.randn(sequence_length,batch_size,model_dimension)
hidden = (torch.randn(lstm_depth,batch_size,model_dimension), torch.randn(lstm_depth,batch_size,model_dimension))

float_lstm=lstm_for_demonstration(model_dimension,model_dimension,lstm_depth)

quantized_lstm=torch.quantization.quantize_dynamic(float_lstm,{nn.LSTM,nn.Linear},dtype=torch.qint8)
print(f'here is the floating point version of this module:\n{float_lstm}')
print(f'and now the quantized version:{quantized_lstm}')



def print_size_of_model(model, label=""):
    torch.save(model.state_dict(),"./tmp/temp.p")
    size=os.path.getsize("./tmp/temp.p")
    print(f"model:{label} ,  size(KB):{size/1e3}")
    os.remove("./tmp/temp.p")
    return size

f=print_size_of_model(float_lstm,"fp32")
q=print_size_of_model(quantized_lstm,"int8")
print("{0:.2f} times smaller".format(f/q))

print("Floating point FP32")
tic=time.time()
float_lstm.forward(inputs,hidden)
print(f"cost time:{time.time()-tic}")
print("Floating point INT8")
tic=time.time()
quantized_lstm.forward(inputs,hidden)
print(f"cost time:{time.time()-tic}")

out1,hidden1=float_lstm(inputs,hidden)
mag1=torch.mean(abs(out1)).item()
print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

out2,hidden2=quantized_lstm(inputs,hidden)
mag2=torch.mean(abs(out2)).item()
print('mean absolute value of output tensor values in the INT8 model is {0:.5f} '.format(mag2))
