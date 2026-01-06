#!/usr/bin/env python3
# shebang line for linux / mac

import os
import signal
import argparse
from dataset import Dataset
from model import ModelFullyconnected, ModelConvNet, ModelConvNet3, ModelBetterCNN
from trainer import Trainer


def sigintHandler(signum, frame):
    print('\nComando recebido. A sair...\n')
    exit(0)


def main():

    ## Definir os argumentos de entrada
    parser = argparse.ArgumentParser()

    parser.add_argument('-df', '--dataset_folder', type=str, default='../mnist')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training and testing.')
    parser.add_argument('-rt', '--resume_training', action='store_true', help='Resume training from last checkpoint if available.')
    parser.add_argument('-ep', '--experiment_path', type=str, default='./experiments', help='Path to save experiment results')

    args = vars(parser.parse_args())
    print(f"\nNº de épocas : {args['num_epochs']}")
    print(f"Tamanho do Batch: {args['batch_size']}")
    print("Treino retomado!\n" if args["resume_training"] else "Novo treino!\n")

    signal.signal(signal.SIGINT, sigintHandler)


    ##  Criar a pasta da experiência
    args['experiment_full_name'] = args['experiment_path']
    os.makedirs(args['experiment_full_name'], exist_ok=True)


    ## Criar os datasets
    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)


    ## Criar o modelo (escolhendo uma das classes em model.py)
    model = ModelBetterCNN()
    # model = ModelConvNet3()
    # model = ModelConvNet()
    # model = ModelFullyconnected()


    ## Iniciar treino
    trainer = Trainer(args, train_dataset, test_dataset, model)

    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    main()
