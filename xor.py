import numpy
import torch
import uniplot


### TODO future idea: make a model that turns any picture into a picture of obama by rearanging the pixels 
### TODO future idea: make a model that turns any picture into a picture of obama by rearanging the pixels 
### TODO future idea: make a model that turns any picture into a picture of obama by rearanging the pixels 
### TODO future idea: make a model that turns any picture into a picture of obama by rearanging the pixels 
### TODO future idea: make a model that turns any picture into a picture of obama by rearanging the pixels 
### TODO future idea: make a model that turns any picture into a picture of obama by rearanging the pixels 

# XOR
## X = Features / Input / Questions
X = torch.tensor([[0,0],[1,1],[0,1],[1,0]], dtype=torch.float32) 
## Y = Labels / Output / Answers
Y = torch.tensor([ [0],  [0],  [1],  [1] ], dtype=torch.float32)

## Non-linear Equation
ACTIVATION  = torch.nn.Sigmoid()
activations = [
    torch.nn.Sigmoid(),
    #torch.nn.ELU(),
    torch.nn.Hardshrink(),
    torch.nn.Hardsigmoid(),
    torch.nn.Hardtanh(),
    torch.nn.Hardswish(),
    torch.nn.LeakyReLU(),
    torch.nn.LogSigmoid(),
    #torch.nn.MultiheadAttention(),
    #torch.nn.PReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU6(),
    torch.nn.RReLU(),
    torch.nn.SELU(),
    torch.nn.CELU(),
    torch.nn.GELU(),
    torch.nn.Sigmoid(),
    torch.nn.SiLU(),
    torch.nn.Mish(),
    torch.nn.Softplus(),
    torch.nn.Softshrink(),
    torch.nn.Softsign(),
    torch.nn.Tanh(),
    torch.nn.Tanhshrink(),
    #torch.nn.Threshold(),
    #torch.nn.GLU(),
    torch.nn.GELU(),
]

class XORModel(torch.nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        units           = 5
        self.input      = torch.nn.Linear(2, units)
        self.input2     = torch.nn.Linear(units, units)
        self.output     = torch.nn.Linear(units, 1)
        self.activation = ACTIVATION

    def forward(self, input):
        out = self.input(input)
        out = self.activation(out)
        out = self.input2(out)
        out = self.activation(out)
        out = self.output(out)
        out = self.activation(out)
        return out
    
model         = XORModel()
delta         = torch.nn.BCELoss()
learning_rate = 0.01
optimizer     = torch.optim.AdamW(model.parameters(), lr=learning_rate)
epochs        = range(1000)
losses        = []

def printAllActivations():
    x = numpy.array(range(-80, 80))
    for activation in activations:
        print(activation)
        y = x / 15
        y = numpy.array(activation(torch.tensor(y, dtype=torch.float32)))
        #print(y)
        uniplot.plot(y, x, title="activation function")

#printAllActivations()


def train():
    for _ in epochs:
        prediction = model(X)
        loss = delta(prediction, Y)
        losses.append(loss.detach().cpu().numpy())
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ##print(torch.round(prediction))

    print(losses)
    uniplot.plot(epochs, losses, title="losses")
train()
