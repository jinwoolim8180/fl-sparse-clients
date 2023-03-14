from argparse import Namespace
from modulefinder import Module
import math
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

def _param_distance(param1, param2):
    dist = 0
    for k in param1.keys():
        dist += (param1[k] - param2[k]).double().norm(2).item()
    return math.sqrt(dist)

class Server:
    def __init__(self, args):
        self.args: Namespace = args
        self.weight_sum = 0
        self.model_parameters: Dict[str, torch.Tensor] = models.get_model(args).state_dict()
        self.lambda_var = torch.zeros(self.args.clients)
        self.client_delta = {}
        self.client_losses = [0 for _ in range(self.args.clients)]
        self.curr_sel_clients = random.sample(range(self.args.clients), self.args.sel_clients)
        self.weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.clients
        self.similarity = torch.eye(self.args.clients)
        self.param_product = torch.zeros(self.args.clients, self.args.clients)

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

            for i in range(self.args.clients):
                self.weights[i] /= sum(self.weights)

            # update delta_avg by updated weights
            print('selection complete.')
            delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
            for client_id in self.client_delta.keys():
                    for k in delta_avg.keys():
                        delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * self.weights[client_id]).type(delta_avg[k].dtype)

        elif self.args.sel_scheme == 'practical':
            sel_clients = []
            param_product = torch.zeros(self.args.clients, self.args.clients)
            norm_num = 0
            norm_avg = 0
            # update parameter products
            for i in self.curr_sel_clients:
                for j in self.curr_sel_clients:
                    param_product[i][j] = _param_dot_product(self.client_delta[i], self.client_delta[j])
                    if i == j:
                        norm_num += 1
                        norm_avg += param_product[i][j]
            norm_avg /= norm_num
            # update similarity matrix
            for i in self.curr_sel_clients:
                for j in self.curr_sel_clients:
                    if i != j:
                        self.similarity[i][j] = param_product[i][j] / (math.sqrt(param_product[i][i]) * math.sqrt(param_product[j][j]))
            for client in range(self.args.sel_clients):
                change = torch.zeros(self.args.clients)
                delta_curr = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
                diff = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
                # update change
                for client_id in self.client_delta.keys():
                    change[client_id] = 0
                    if client_id not in sel_clients:
                        for i in self.client_delta.keys():
                            if client_id in self.curr_sel_clients and i in self.curr_sel_clients:
                                change[client_id] += (1 / self.args.clients - self.weights[i]) * param_product[client_id][i]
                            else:
                                similarity = random.uniform(-0.5, 0.8) if self.similarity[client_id][i] == 0 else self.similarity[client_id][i]
                                if client_id in self.curr_sel_clients:
                                    change[client_id] += (1 / self.args.clients - self.weights[i]) * 0.5 * (param_product[client_id][client_id] + norm_avg) * similarity
                                elif i in self.curr_sel_clients:
                                    change[client_id] += (1 / self.args.clients - self.weights[i]) * 0.5 * (param_product[i][i] + norm_avg) * similarity
                                else:
                                    change[client_id] += (1 / self.args.clients - self.weights[i]) * norm_avg * similarity
                        change[client_id] += self.args.lambda_loss * self.client_losses[client_id]
                sel_client = np.random.choice(torch.where(change == torch.amax(change))[0].cpu().numpy())
                sel_clients.append(sel_client) # argmax needs to be changed
                # update weight through gradient descent
                self.weights = [1/(client + 1) if i in sel_clients else 0 for i in range(self.args.clients)]
                for iter in range(self.args.iteration):
                    # update weight
                    for client_id in sel_clients:
                        for i in self.client_delta.keys():
                            if client_id in self.curr_sel_clients and i in self.curr_sel_clients:
                                self.weights[client_id] += (1 / self.args.clients - self.weights[i]) * param_product[client_id][i]
                            else:
                                similarity = random.uniform(-0.5, 0.8) if self.similarity[client_id][i] == 0 else self.similarity[client_id][i]
                                if client_id in self.curr_sel_clients:
                                    self.weights[client_id] += (1 / self.args.clients - self.weights[i]) * 0.5 * (param_product[client_id][client_id] + norm_avg) * similarity
                                elif i in self.curr_sel_clients:
                                    self.weights[client_id] += (1 / self.args.clients - self.weights[i]) * 0.5 * (param_product[i][i] + norm_avg) * similarity
                                else:
                                    self.weights[client_id] += (1 / self.args.clients - self.weights[i]) * norm_avg * similarity
                        self.weights[client_id] += self.args.lambda_loss * self.client_losses[client_id]

            # update delta_avg by updated weights
            print('selection complete.')
            self.curr_sel_clients = sel_clients
            delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
            for client_id in self.client_delta.keys():
                    for k in delta_avg.keys():
                        delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * self.weights[client_id]).type(delta_avg[k].dtype)
                        
        elif self.args.sel_scheme == 'divfl':
            sel_clients = []
            list_clients = list(range(self.args.clients))
            dist_matrix = torch.zeros(self.args.clients, self.args.clients)
            for i in range(self.args.clients):
                for j in range(self.args.clients):
                    dist_matrix[i][j] = _param_distance(self.client_delta[i], self.client_delta[j])
            for iter in range(self.args.sel_clients):
                sel_client = 0
                min_score = 1000
                for kk in list_clients:
                    G_ks = 0
                    for k in range(self.args.clients):
                        min_dist = 100
                        for i in sel_clients:
                            dist = dist_matrix[k][i]
                            min_dist = dist if dist < min_dist else min_dist
                        dist = dist_matrix[k][kk]
                        min_dist = dist if dist < min_dist else min_dist
                        G_ks += min_dist
                    if G_ks < min_score:
                        min_score = G_ks
                        sel_client = kk
                sel_clients.append(sel_client)
                list_clients.remove(sel_client)
            # update delta_avg by updated weights
            print('selection complete.')
            delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
            weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.sel_clients
            for client_id in sel_clients:
                    for k in delta_avg.keys():
                        delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(delta_avg[k].dtype)

        elif self.args.sel_scheme == 'power-of-choice':
            client_losses = np.array(self.client_losses)
            topk = np.argpartition(client_losses, -self.args.sel_clients)[-self.args.sel_clients:]
            weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.sel_clients
            for client_id in topk:
                for k in delta_avg.keys():
                    delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(delta_avg[k].dtype)

        elif self.args.sel_scheme == 'random':
            topk = np.random.permutation(self.args.clients)[-self.args.sel_clients:]
            weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.sel_clients
            for client_id in topk:
                for k in delta_avg.keys():
                    delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(delta_avg[k].dtype)

        for k in self.model_parameters.keys():
            self.model_parameters[k].add_(delta_avg[k])

        if self.args.climb:
            client_losses = torch.tensor(self.client_losses)
            self.lambda_var += self.args.lambda_lr * (client_losses - torch.mean(client_losses) - self.args.epsilon) / self.args.clients
            self.lambda_var = torch.clamp(self.lambda_var, min=0., max=100.)

        self.client_delta = {}
        self.weight_sum = 0