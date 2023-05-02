import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import logging
import wandb



class EvaluationMetric(object):
    """
    Computes and stores statistics and evaluation metrics.
    """

    def __init__(self, number_of_classes, INVERSE_EVENT_DICTIONARY):
        self.number_of_classes = number_of_classes
        self.classes = torch.zeros(number_of_classes)
        self.classes_ground = torch.zeros(number_of_classes)
        self.loss_total = 0
        self.total = 0
        self.classes_names = []
        self.attention_sum = [0, 0]
        self.total_attention = 0
        for i in range(number_of_classes):
            self.classes_names.append(INVERSE_EVENT_DICTIONARY[i])

        self.total_predictions = []
        self.total_groundTruth = []
        
    def reset(self):
        self.classes = torch.zeros(self.number_of_classes)
        self.classes_sec = torch.zeros(self.number_of_classes)
        self.classes_ground = torch.zeros(self.number_of_classes)
        self.loss_total = 0
        self.total = 0
        self.total_attention = 0
        self.attention_sum = [0, 0]
        self.total_predictions = []
        self.total_groundTruth = []

    def update(self, loss, outputs, targets, attention):
        self.loss_total += float(loss)
        self.total += 1
    
        if attention.size()[0] == 2:
            self.total_attention += attention.size()[0]
            self.attention_sum[0] += torch.sum(attention, dim=0)[0]
        
        preds = torch.argmax(outputs, 1)
        preds_sec = torch.topk(outputs, 2)
        preds_sec = preds_sec[1][:, 1]

        grountruths = torch.argmax(targets, 1)
        aux = preds == grountruths
        aux2 = preds_sec == grountruths

        for i in range(len(aux)):
            if aux[i]:
                self.classes[grountruths[i]] += 1
                self.classes_sec[grountruths[i]] += 1
            if aux2[i]:
                self.classes_sec[grountruths[i]] += 1
            self.classes_ground[grountruths[i]] += 1

    def get_metrics(self, epoch):
        if self.total == 0:
            loss_avg = 0
        else:
            loss_avg = self.loss_total / self.total

        if sum(self.classes_ground) == 0:
            acc = 0
        else:
            acc = sum(self.classes) / sum(self.classes_ground)

        if sum(self.classes_ground) == 0:
            acc2 = 0
        else:
            acc2 = sum(self.classes_sec) / sum(self.classes_ground)


        if self.total_attention != 0:
            attention_aux = self.attention_sum[0] / self.total_attention
            attention_aux = attention_aux.item()
        else:
            attention_aux = -1

        return loss_avg, acc.item(), acc2.item(), self.classes, self.classes_sec, self.classes_ground, self.total_predictions, self.total_groundTruth, self.classes_names, attention_aux

