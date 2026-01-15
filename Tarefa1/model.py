#!/usr/bin/env python3
# shebang line for linux / mac

from torchinfo import summary
import torch.nn as nn


class ModelFullyconnected(nn.Module):

    def __init__(self):
        super(ModelFullyconnected, self).__init__()  # call the parent constructor

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define the layers of the model
        self.fc = nn.Linear(ninputs, noutputs)

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def forward(self, x):

        # flatten the input to a vector of 1x28x28
        x = x.view(x.size(0), -1)

        # Now we can pass through the fully connected layer
        y = self.fc(x)

        return y

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelConvNet(nn.Module):

    def __init__(self):

        super(ModelConvNet, self).__init__()

        # Define first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 32x14x14

        # Define second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # this will output 64x14x14

        # Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define the first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # this will output 128

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # this will output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        print('Forward method called ...')
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # Transform to latent vector
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        y = self.fc2(x)

        return y


class ModelConvNet3(nn.Module):
    """This is a more complex ConvNet model with 3 conv layers."""

    def __init__(self):

        super(ModelConvNet3, self).__init__()

        # Define first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 32x14x14

        # Define second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # this will output 64x14x14

        # Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define second conv layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # this will output 128x4x4

        # Define the second pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 128x2x2

        # Define the first fully connected layer
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        # this will output 128

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # this will output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # Transform to latent vector
        x = x.view(-1, 128*2*2)

        x = self.fc1(x)

        y = self.fc2(x)

        return y


## Nova arquitetura de modelo CNN
class ModelBetterCNN(nn.Module):

    def __init__(self):
        super(ModelBetterCNN, self).__init__()

        ## Bloco de extração de características
        self.features = nn.Sequential(
            
            ## Bloco 1: 1x28x28 -> 32x28x28 -> pool -> 32x14x14

            ## Primeira convolução: aumenta a profundidade de 1 para 32 canais
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),      ## Normalização para estabilizar o treino
            nn.ReLU(inplace=True),   ## Função de ativação não linear para aprender relações complexas
            ## a saída é 32x28x28

            ## Segunda convolução: refina as características extraídas
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ## a saída é 32x28x28

            ## Redução da dimensão espacial (28x28 -> 14x14)
            nn.MaxPool2d(kernel_size=2, stride=2),

            ## Dropout para regularização
            nn.Dropout2d(p=0.25),


            ## Bloco 2: 32x14x14 -> 64x14x14 -> pool -> 64x7x7

            ## Primeira camada de convulção
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ## a saída é 64x14x14

            ## Segunda camada de convulção
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ## a saída é 64x14x14

            ## Nova redução da dimensão espacial (14x14 -> 7x7)
            nn.MaxPool2d(kernel_size=2, stride=2),

            ## Dropout para regularização
            nn.Dropout2d(p=0.25),
        )

        ## Bloco de classificação
        self.classifier = nn.Sequential(
            ## Converte os mapas de características num vetor 1D
            nn.Flatten(),

            ## Primeira camada fully connected
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            ## Segunda camada fully connected
            nn.Linear(256, 10),
            ## saída 10
        )

        print('Arquitetura do modelo inicializada com ' + str(self.getNumberOfParameters()) + ' parâmetros.')
        # summary(self, input_size=(1, 1, 28, 28))

    def forward(self, x):
        x = self.features(x)
        y = self.classifier(x)
        return y

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

