from argparse import Namespace
from modulefinder import Module
import random
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

import models

def _param_dot_product(param1, param2):
    product = 0
    for k in param1.keys():
        product += torch.sum(param1[k] * param2[k])
    return product

class Server:
    def __init__(self, args):
        self.args: Namespace = args
        self.weight_sum = 0
        self.model_parameters: Dict[str, torch.Tensor] = models.get_model(args).state_dict()
        self.lambda_var = torch.zeros(self.args.clients)
        self.client_delta = {}
        self.client_losses = [0 for _ in range(self.args.clients)]
        self.sel_clients = []
        self.weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.clients
        self.similarity = torch.zeros(self.args.clients, self.args.clients)

    def update_client_param(self, client_id, client_delta, weight, loss):
        self.weight_sum += weight
        self.client_losses[client_id] = loss
        self.client_delta[client_id] = client_delta

    def aggregate(self):
        weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.clients
        delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
        
        for client_id in self.client_delta.keys():
            for k in delta_avg.keys():
                delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(delta_avg[k].dtype)

        if self.args.sel_scheme == 'ideal':
            sel_clients = []
            for client in range(self.args.sel_clients):
                change = torch.zeros(self.args.clients)
                delta_curr = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
                diff = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
                # current aggregated gradient
                for client_id in self.client_delta.keys():
                    for k in delta_curr.keys():
                        delta_curr[k] = delta_curr[k] + (self.client_delta[client_id][k] * self.weights[client_id]).type(delta_curr[k].dtype)
                # current difference from global gradient
                for k in diff.keys():
                    diff[k] = delta_avg[k] - delta_curr[k]
                # update change
                for client_id in self.client_delta.keys():
                    change[client_id] = 0 if client_id in sel_clients else _param_dot_product(self.client_delta[client_id], diff) + self.args.lambda_loss * self.client_losses[client_id]
                sel_clients.append(torch.argmax(change).item())
                # update weight through gradient descent
                self.weights = [1/(client + 1) if i in sel_clients else 0 for i in range(self.args.clients)]
                for iter in range(self.args.iteration):
                    delta_curr = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
                    diff = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
                    # current aggregated gradient
                    for client_id in self.client_delta.keys():
                        for k in delta_curr.keys():
                            delta_curr[k] = delta_curr[k] + (self.client_delta[client_id][k] * self.weights[client_id]).type(delta_curr[k].dtype)
                    # current difference from global gradient
                    for k in diff.keys():
                        diff[k] = delta_avg[k] - delta_curr[k]
                    # update weight
                    for client_id in sel_clients:
                        self.weights[client_id] += _param_dot_product(self.client_delta[client_id], diff) + self.args.lambda_loss * self.client_losses[client_id]

            # update delta_avg by updated weights
            print('selection complete.')
            delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
            for client_id in self.client_delta.keys():
                    for k in delta_avg.keys():
                        delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * self.weights[client_id]).type(delta_avg[k].dtype)

        elif self.args.sel_scheme == 'power-of-choice':
            client_losses = np.array(self.client_losses)
            topk = np.argpartition(client_losses, -self.args.sel_clients)[-self.args.sel_clients:]
            # topk = np.random.permutation(self.args.clients)[:clients]
            for client_id in topk:
                for k in delta_avg.keys():
                    delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(delta_avg[k].dtype)

        elif self.args.sel_scheme == 'random':
            topk = np.argpartition(np.array(range(self.args.clients)), -self.args.sel_clients)[-self.args.sel_clients:]
            for client_id in topk:
                for k in self.delta_avg.keys():
                    self.delta_avg[k] = self.delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(self.delta_avg[k].dtype)

        for k in self.model_parameters.keys():
            self.model_parameters[k].add_(delta_avg[k])

        if self.args.climb:
            client_losses = torch.tensor(self.client_losses)
            self.lambda_var += self.args.lambda_lr * (client_losses - torch.mean(client_losses) - self.args.epsilon) / self.args.clients
            self.lambda_var = torch.clamp(self.lambda_var, min=0., max=100.)

        self.client_delta = {}
        self.weight_sum = 0