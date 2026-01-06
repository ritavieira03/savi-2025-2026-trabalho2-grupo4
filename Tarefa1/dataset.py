import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):

        ## Armazenar os argumentos nas propriedades da classe
        self.args = args
        self.train = is_train


        ## Criar os inputs
        split_name = 'train' if is_train else 'test'
        image_path = os.path.join(args['dataset_folder'], split_name, 'images/')
        self.image_filenames = glob.glob(image_path + "/*.jpg")
        
        ## Ordenar os nomes dos ficheiros para garantir uma ordem consistente
        self.image_filenames.sort()

        ## Criar as labels
        self.labels_filename = os.path.join(
            args['dataset_folder'], split_name, 'labels.txt')

        self.labels = []

        with open(self.labels_filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                label = float(parts[1])
                self.labels.append(label)


        ## selecionar a percentagem de exemplos a utilizar
        num_examples = round(len(self.image_filenames) * 0.1)

        ## reduz o tamanho das image_fileanames e das labels
        self.image_filenames = self.image_filenames[0:num_examples]
        self.labels = self.labels[0:num_examples]


        ## Converter de uma lista para um tensor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        ## Esta função retorna o número de exemplos no dataset
        return len(self.image_filenames)

    def __getitem__(self, idx):
        ## Esta função recebe como input o idx de um exemplo e deve devolver o input e o output correspondente a esse exemplo.
        ## retorna (image_tensor, label_tensor)


        ## Obter a label como um tensor
        label_index = int(self.labels[idx])
        label = [0]*10  # create a list of ten zeros
        label[label_index] = 1  # set the position of the label to 1

        label_tensor = torch.tensor(label, dtype=torch.float)

        ## Obter a imagem como um tensor
        image_filename = self.image_filenames[idx]

        image = Image.open(image_filename).convert('L')  # grayscale
        image_tensor = self.to_tensor(image)

        return image_tensor, label_tensor