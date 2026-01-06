import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model

        # Create the dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        # For testing we typically set shuffle to false

        # Setup loss function
        self.loss = nn.MSELoss()  # Mean Squared Error Loss

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        # Start from scratch or resume training
        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def train(self):

        print('Training started.  Max epochs = ' + str(self.args['num_epochs']))

        # -----------------------------------------
        # Iterate all epochs
        # -----------------------------------------
        for i in range(self.epoch_idx, self.args['num_epochs']):  # number of epochs

            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))
            
            # -----------------------------------------
            # Train - Iterate over batches
            # -----------------------------------------
            self.model.train()  # set model to training mode
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):  # type: ignore

                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # Update model
                self.optimizer.zero_grad()  # resets the gradients from previous batches
                batch_loss.backward()  # the actual backpropagation
                self.optimizer.step()

            # -----------------------------------------
            #  Test - Iterate over batches
            # -----------------------------------------
            self.model.eval()  # set model to evaluation mode

            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):  # type: ignore
                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # During test there is no model update

            # ---------------------------------
            #  End of the epoch training
            # ---------------------------------
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']-1))
            # print('batch_losses: ' + str(batch_losses))

            # update the training epoch losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            # update the testing epoch losses
            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            # Draw the updated training figure
            self.draw()

            # Save the training state
            self.saveTrain()

        print('Training completed.')
        print('Training losses: ' + str(self.train_epoch_losses))
        print('Test losses: ' + str(self.test_epoch_losses))

    
    def loadTrain(self):
        print('Resuming training from last checkpoint.')

        # find the checkpoint file
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        print('checkpoint_file: ' + str(checkpoint_file))

        # Verify if file exists. If not abort. Cannot resume without the checkpoint.pkl
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        print(checkpoint.keys())

        self.epoch_idx = checkpoint['epoch_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])  # contains the model's weights
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])  # contains the optimizer's


    def saveTrain(self):

        # Create the dictionary to save the checkpoint.pkl
        checkpoint = {}
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['test_epoch_losses'] = self.test_epoch_losses

        checkpoint['model_state_dict'] = self.model.state_dict()  # contains the model's weights
        # contains the optimizer's state
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

        # Save the best.pkl
        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            best_file = os.path.join(self.args['experiment_full_name'], 'best.pkl')
            torch.save(checkpoint, best_file)

    def draw(self):

        plt.figure(1)  # creates a new fig therefore clears all past drawings
        plt.clf()

        # Setup the figure
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])  # type: ignore
        axis.set_ylim([0, 0.1])  # type: ignore

        # plot training
        xs = range(1, len(self.train_epoch_losses)+1)
        ys = self.train_epoch_losses
        plt.plot(xs, ys, 'r-', linewidth=2)

        # plot testing
        xs = range(1, len(self.test_epoch_losses)+1)
        ys = self.test_epoch_losses
        plt.plot(xs, ys, 'b-', linewidth=2)

        # draw best checkpoint
        best_epoch_idx = int(np.argmin(self.test_epoch_losses))
        print('best_epoch_idx: ' + str(best_epoch_idx))
        plt.plot([best_epoch_idx, best_epoch_idx], [0, 0.5], 'g--', linewidth=1)

        plt.legend(['Train', 'Test', 'Best'], loc='upper right')

        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))


    def evaluate(self):

<<<<<<< HEAD
        self.model.eval()
        print('\nEvaliação do modelo')
=======
        self.model.eval()  # modo avaliação
        print('\nAvaliação do modelo')
>>>>>>> 090d5ada884631e01814c00a0e70435f09e04146

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
        # print("Confusion Matrix:\n", cm)

        ## Accuracy
        accuracy = accuracy_score(gt_classes, predicted_classes)
        print(f"\nTest Accuracy: {accuracy*100:.4f}%")

        ## Precision, Recall e F1-Score
        precision_per_class = precision_score(gt_classes, predicted_classes, labels=range(10), average=None)
        recall_per_class    = recall_score(gt_classes, predicted_classes, labels=range(10), average=None)
        f1_per_class        = f1_score(gt_classes, predicted_classes, labels=range(10), average=None)

        ## Média Macro
        precision_macro = precision_score(gt_classes, predicted_classes, labels=range(10), average='macro')
        recall_macro    = recall_score(gt_classes, predicted_classes, labels=range(10), average='macro')
        f1_macro        = f1_score(gt_classes, predicted_classes, labels=range(10), average='macro')

        print(Style.BRIGHT + "\nMetrics per class:" + Style.RESET_ALL)
        for i in range(10):
            print(f"Classe {i}: Precision = {precision_per_class[i]:.4f}, Recall = {recall_per_class[i]:.4f}, F1 = {f1_per_class[i]:.4f}")

        print(f"\nMédia Macro: Precision = {precision_macro:.4f}, Recall = {recall_macro:.4f}, F1 = {f1_macro:.4f}")

        ## Guardar os reultados em JSON
        metrics_dict = {}
        for i in range(10):
            metrics_dict[i] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }

        metrics_dict['macro_average'] = {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro),
            'accuracy': float(accuracy)
        }

        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(metrics_dict, f, indent=4)


        ## Desenhar a matriz de confusão em PNG
        self.plot_confusion_matrix(cm)

        ## Desenhar a tabela de resultados em PNG
        self.save_metrics_table_png(precision_per_class, recall_per_class, f1_per_class,
                                    precision_macro, recall_macro, f1_macro)


    def save_metrics_table_png(self,
    precision_per_class, recall_per_class, f1_per_class,
    precision_macro, recall_macro, f1_macro):
    
        # Número de classes
        n_classes = len(precision_per_class)

        # Colunas
        col_labels = ["Precision", "Recall", "F1_score"]

        # Linhas
        row_labels = [str(i) for i in range(n_classes)] + ["Média Macro"]

        cell_text = []
        for i in range(n_classes):
            cell_text.append([
                f"{precision_per_class[i]:.3f}",
                f"{recall_per_class[i]:.3f}",
                f"{f1_per_class[i]:.3f}"
            ])

        cell_text.append([
            f"{precision_macro:.3f}",
            f"{recall_macro:.3f}",
            f"{f1_macro:.3f}"
        ])

        # Figura
        fig_h = 0.55 * len(row_labels) + 1.0
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.0, 1.6)

        azul_clarinho = "#dbeafe"  # azul bem clarinho

        # Estilo geral da tabela
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.2)

            # Estilo das colunas
            if r == 0:
                cell.set_facecolor(azul_clarinho)
                cell.set_text_props(weight="bold")

            # Estilo das linhas
            if c == 0:
                cell.set_facecolor(azul_clarinho)
                cell.set_text_props(weight="bold")

        plt.tight_layout()

        out_path = os.path.join(self.args["experiment_full_name"], "results_table.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)


    def plot_confusion_matrix(self, cm):
        
        plt.figure(2)
        class_names = [str(i) for i in range(10)]
        title = 'Confusion Matrix'
        seaborn.heatmap(cm,
                        annot=True,       # Anotar as células com os valores
                        fmt='d',          # Formato dos números (inteiros para contagens)
                        # Mapa de cores (pode escolher outro, ex: 'viridis', 'YlGnBu')
                        cmap='Blues',
                        cbar=True,        # Mostrar barra de cores
                        xticklabels=class_names,  # Rótulos do eixo X (classes previstas)
                        yticklabels=class_names)  # Rótulos do eixo Y (classes verdadeiras)

        plt.title(title, fontsize=16)  # Título do gráfico
        plt.xlabel('Predicted classes', fontsize=14)  # Rótulo do eixo X
        plt.ylabel('True classes', fontsize=14)  # Rótulo do eixo Y
        plt.xticks(rotation=0, ha='right', fontsize=12)  # Rodar rótulos do X para melhor leitura
        plt.yticks(rotation=0, fontsize=12)  # Rótulos do Y
        plt.tight_layout()  # Ajusta o layout para evitar sobreposições

        plt.savefig(os.path.join(self.args['experiment_full_name'],
                                 'confusion_matrix.png'))
        plt.close()