#!/usr/bin/env python3
# shebang line for linux / mac

import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
from colorama import Style
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        self.args = args
        self.model = model

        ## Criar os dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
    
        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        ## Início ou retoma de treino
        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    
    def train(self):

        print('\nInício do Treino!\nNº máximo de épocas = ' + str(self.args['num_epochs']))

        for i in range(self.epoch_idx, self.args['num_epochs']):

            self.epoch_idx = i
            print('\nÉpoca ' + str(self.epoch_idx))
            
            self.model.train()
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):  # type: ignore

                ## Calcular os predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                ## Calcular as probabilidades, usando softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                ## Calcular as losses, usando MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())

                ## Atualizar o modelo
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            
            self.model.eval()

            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):  # type: ignore
                ## Calcular os predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                ## Calcular as probabilidades, usando softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                ## Calcular o loss usando MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())
                

            print('Época terminada: ' + str(i) + ' de ' + str(self.args['num_epochs']-1))
            
            ## Atualizar o training epoch losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            ## Atualizar o testing epoch losses
            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            ## Desenhar a figura de treino atualizada
            self.draw()

            ## Guardar o estado do treino
            self.saveTrain()


        print(Style.BRIGHT + '\nTreino Completo!' + Style.RESET_ALL)
        print('\nTraining losses: ' + str(self.train_epoch_losses))
        print('\nTest losses: ' + str(self.test_epoch_losses))

    
    def loadTrain(self):
        print('\nRetomar o último treino disponível.')

        # Encontar o ficheiro checkpoint
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')

        # Verificar se o ficheiro existe. Se não existir abortar, não é possível retomar sem o checkpoint.pkl
        if not os.path.exists(checkpoint_file):
            raise ValueError('Ficheiro Checkpoint.pkl não encontrado: ' + checkpoint_file)

        # Carregar o checkpoint
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        print(checkpoint.keys())

        self.epoch_idx = checkpoint['epoch_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])


    def saveTrain(self):

        # Criar o dicionário para salvar o checkpoint.pkl
        checkpoint = {}
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['test_epoch_losses'] = self.test_epoch_losses

        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

        # Guardar best.pkl
        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            best_file = os.path.join(self.args['experiment_full_name'], 'best.pkl')
            torch.save(checkpoint, best_file)

    
    def draw(self):

        ## Desenhar a figura de treino
        plt.figure(1)
        plt.clf()

        ## Labels da figura
        plt.title("Training Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])  # type: ignore
        axis.set_ylim([0, 0.1])  # type: ignore

        ## Gráfico de treino
        xs = range(1, len(self.train_epoch_losses)+1)
        ys = self.train_epoch_losses
        plt.plot(xs, ys, 'r-', linewidth=2)

        ## Gráfico de testing
        xs = range(1, len(self.test_epoch_losses)+1)
        ys = self.test_epoch_losses
        plt.plot(xs, ys, 'b-', linewidth=2)

        ## Desenhar o melhor checkpoint
        best_epoch_idx = int(np.argmin(self.test_epoch_losses))
        print('Melhor época: ' + str(best_epoch_idx))
        plt.plot([best_epoch_idx, best_epoch_idx], [0, 0.5], 'g--', linewidth=1)

        plt.legend(['Train', 'Test', 'Best'], loc='upper right')

        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))

    
    def evaluate(self):

        self.model.eval()
        print(Style.BRIGHT + '\nAvaliação do modelo:' + Style.RESET_ALL)

        gt_classes = []
        predicted_classes = []

        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                enumerate(self.test_dataloader), total=len(self.test_dataloader)):

            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

            label_pred_tensor = self.model.forward(image_tensor)
            label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
            batch_predicted_classes = label_pred_probabilities_tensor.argmax(dim=1).tolist()

            gt_classes.extend(batch_gt_classes)
            predicted_classes.extend(batch_predicted_classes)

        gt_classes = np.array(gt_classes)
        predicted_classes = np.array(predicted_classes)

        ## Matriz de Confusão
        cm = confusion_matrix(gt_classes, predicted_classes)

        ## Accuracy
        accuracy = accuracy_score(gt_classes, predicted_classes)
        print(Style.BRIGHT + f"\nAccuracy:" + Style.RESET_ALL + f" {accuracy*100:.4f}%")

        ## Precision, Recall e F1 Score
        precision_per_class = precision_score(gt_classes, predicted_classes, labels=range(10), average=None)
        recall_per_class    = recall_score(gt_classes, predicted_classes, labels=range(10), average=None)
        f1_per_class        = f1_score(gt_classes, predicted_classes, labels=range(10), average=None)

        ## Média Macro
        precision_macro = precision_score(gt_classes, predicted_classes, labels=range(10), average='macro')
        recall_macro    = recall_score(gt_classes, predicted_classes, labels=range(10), average='macro')
        f1_macro        = f1_score(gt_classes, predicted_classes, labels=range(10), average='macro')

        print(Style.BRIGHT + "\nResultados:" + Style.RESET_ALL)
        for i in range(10):
            print(f"Classe {i}: Precision = {precision_per_class[i]:.4f}, Recall = {recall_per_class[i]:.4f}, F1 = {f1_per_class[i]:.4f}")

        print(f"\nMédia Macro: Precision = {precision_macro:.4f}, Recall = {recall_macro:.4f}, F1 = {f1_macro:.4f}\n")


        ## Guardar os resultados em JSON
        self.save_results_json(precision_per_class, recall_per_class, f1_per_class,
                               precision_macro, recall_macro, f1_macro, accuracy)

        ## Desenhar a matriz de confusão, em png
        self.plot_confusion_matrix(cm)

        ## Desenhar a tabela de resultados, em png
        self.save_results_table(precision_per_class, recall_per_class, f1_per_class,
                                precision_macro, recall_macro, f1_macro)

    
    ## Função para guardar os resultados em JSON
    def save_results_json(self, precision_per_class, recall_per_class, f1_per_class,
                                    precision_macro, recall_macro, f1_macro, accuracy):
        results_dict = {}
        for i in range(10):
            results_dict[i] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }

        results_dict['macro_average'] = {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro),
            'accuracy': float(accuracy)
        }

        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(results_dict, f, indent=4)

    ## Função para desenhar a tabela de resultados
    def save_results_table(self, precision_per_class, recall_per_class, f1_per_class,
                                     precision_macro, recall_macro, f1_macro):
    
        ## Número de classes
        n_classes = len(precision_per_class)

        ## Colunas
        col_labels = ["Precision", "Recall", "F1 Score"]

        ## Linhas
        row_labels = [str(i) for i in range(n_classes)] + ["Média Macro"]

        cell_text = []
        for i in range(n_classes):
            cell_text.append([
                f"{precision_per_class[i]:.4f}",
                f"{recall_per_class[i]:.4f}",
                f"{f1_per_class[i]:.4f}"
            ])

        cell_text.append([
            f"{precision_macro:.4f}",
            f"{recall_macro:.4f}",
            f"{f1_macro:.4f}"
        ])

        ## Figura
        fig_h = 0.55 * len(row_labels)+ 0.5
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="center",
            loc="center"
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.0, 1.6)

        ## Estilo geral da tabela
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.2)

            ## Estilo das colunas
            if r == 0:
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(weight="bold")

            ## Estilo das linhas
            if c == -1:
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(weight="bold")

            if r in (0, len(row_labels)) or r == n_classes+1 or c in (-1, len(col_labels)):
                cell.set_linewidth(1.75)

        plt.tight_layout()

        out_path = os.path.join(self.args["experiment_full_name"], "results_table.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    ## Função para desenhar a matriz de confusão
    def plot_confusion_matrix(self, cm):
        
        plt.figure(2)
        for i in range(10):
            class_names = [str(i)]
        
        title = 'Confusion Matrix'
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=class_names, yticklabels=class_names)

        plt.title(title, fontsize=16)
        plt.xlabel('Predicted classes', fontsize=14)
        plt.ylabel('True classes', fontsize=14)
        plt.xticks(rotation=0, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()

        plt.savefig(os.path.join(self.args['experiment_full_name'],'confusion_matrix.png'))
        plt.close()