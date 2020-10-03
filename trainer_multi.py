import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
import numpy as np

import torch.nn as nn

from utils import _create_model_training_folder

NUM_CLASS = 10
device = torch.device('cuda')


def loss_function(prob, label):
    loss = -torch.log(prob.gather(1, label.unsqueeze(1))+ 1e-8)
    trg = label[:,None]
    one_hot_target = (trg == torch.arange(NUM_CLASS).to(device).reshape(1, NUM_CLASS)).float().bool()
    reg_loss = torch.log(1 - prob + 1e-8) * (~one_hot_target)
    reg_loss = reg_loss.sum()
    return torch.sum(loss) - reg_loss

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, model_path, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.model_path = model_path
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset, flag_ova):
        
        if flag_ova:
            criterion = loss_function
        else:
            criterion = nn.CrossEntropyLoss()

        valid_dataset = Subset(train_dataset[1], np.arange(int(len(train_dataset[1]) * 0.8), len(train_dataset[1])))
        train_dataset = Subset(train_dataset[0], np.arange(0, int(len(train_dataset[0]) * 0.8)))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)


        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        max_acc = None
        for epoch_counter in range(self.max_epochs):

            for (batch_view_1, batch_view_2), labels in train_loader:
                
                # print(batch_view_1.shape)
                # print(batch_view_2.shape)
                # print(labels.shape)

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                labels = labels.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                byol_loss, org_loss, loss = self.update(batch_view_1, batch_view_2, labels, criterion)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
            
            print("End of epoch {}, Byol Loss {:.4f}, Org Loss {:.4f}, Total Loss {:.4f}  ".format(epoch_counter, byol_loss, org_loss, loss))
#             print("End of epoch {}".format(epoch_counter))

            total = 0
            correct = 0
            with torch.no_grad():
                self.online_network.eval()
                val_loss = 0
                for i, data in enumerate(valid_loader, 0):
                    inputs, labels = data
                    inputs_1 = inputs[0].to(device)
                    inputs_2 = inputs[1].to(device)
                    labels = labels.to(device)

                    _, outputs_1 = self.online_network(inputs_1)
                    _, outputs_2 = self.online_network(inputs_2)
                    loss_1 = criterion(outputs_1, labels)
                    loss_2 = criterion(outputs_2, labels)
                    loss = (loss_1 + loss_2) / 2
                    val_loss += loss.item()

                    outputs = (outputs_1 + outputs_2) / 2

                    predicted_value, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                acc = 100 * correct / total
                val_loss = val_loss / len(valid_loader)

                print('Epoch [{}/{}]: Val Accuracy: {}, Val pred Loss: {} '.format(epoch_counter + 1, self.max_epochs,
                                                                                   acc, val_loss))

            if max_acc is None:
                max_acc = acc
            if acc > max_acc:
                print("saved!")
                max_acc = acc
                # save checkpoints
                self.save_model(self.model_path)

    def update(self, batch_view_1, batch_view_2, labels, criterion):
        self.online_network.train()
        proj1, out1 = self.online_network(batch_view_1)
        proj2, out2 = self.online_network(batch_view_2)
        
        org_loss_1 = criterion(out1, labels)
        org_loss_2 = criterion(out2, labels)
        org_loss = (org_loss_1 + org_loss_2) / 2
        
        # compute query feature
        predictions_from_view_1 = self.predictor(proj1)
        predictions_from_view_2 = self.predictor(proj2)

        # compute key features
        with torch.no_grad():
            targets_to_view_2, _ = self.target_network(batch_view_1)
            targets_to_view_1, _ = self.target_network(batch_view_2)

        byol_loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        byol_loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        byol_loss = byol_loss.mean()
        
        loss = org_loss + byol_loss
        
        return byol_loss, org_loss, loss

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
