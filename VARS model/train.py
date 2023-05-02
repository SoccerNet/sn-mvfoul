import logging
import os
from evaluate import EvaluationMetric
import time
from tqdm import tqdm
import torch
import numpy as np
import math
import torch.nn as nn
import wandb
import gc
from datetime import datetime
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY


def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            scheduler_step,
            lr,
            lr_warmup,
            criterion,
            metric_calculator,
            best_model_path,
            epoch_start,
            model_name,
            max_epochs=1000,
            ):
    

    logging.info("start training")

    # counting losses
    counter = 0
    save_tests = []
    best_epoch = 0

    if lr_warmup != 0:
        learning_rate = 0.0000001
        gamma = (lr / learning_rate) ** (1 / lr_warmup)
        lr_warmup += 1
        optimizer.param_groups[0]['lr'] = learning_rate
        scheduler_warmup = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) 

    best_loss = 9e99
    best_metric = -1

    if "gamma" in scheduler.state_dict():
        gamma_aux = scheduler.state_dict()['gamma']
    else:
        gamma_aux = 0

    for epoch in range(epoch_start, max_epochs):

        start = time.time()

        ###################### TRAINING ###################
        metric_training = train(
            train_loader,
            model,
            criterion,
            metric_calculator,
            optimizer,
            epoch + 1,
            train=True,
        )

        # Log to Wandb for TRAINING
        total_loss_training = 0
        for i in range(len(metric_training)):
            if i == 0:
                label = 'offence_severity_class'
            elif i == 1:
                label = 'action_class'

            loss_training, acc_training, acc2_training, classes_training,_, classes_ground_training,_,_,_, attention_training = metric_training[i].get_metrics(epoch+1)
            total_loss_training += loss_training                                                     
            logging.info(label + " training loss at epoch " + str(epoch+1) + " -> " + str(loss_training))
            logging.info(label + " training accuracy at epoch " + str(epoch+1) + " -> " + str(acc_training))
        logging.info("Total training loss at epoch " + str(epoch+1) + " -> " + str(total_loss_training))  
        

        ###################### VALIDATION ALL VIEWS ###################
        metric_vali_2 = train(
            val_loader2,
            model,
            criterion,
            metric_calculator,
            optimizer,
            epoch + 1,
            train = False)

        # LOG INFORMATION OF VALIDATION 1 AND 2
        total_loss_validation = 0
        for i in range(len(metric_vali_2)):
            if i == 0:
                label = 'offence_severity_class'
            elif i == 1:
                label = 'action_class'

            loss_validation2, acc_validation2, acc2_validation2, classes_validation2, _, classes_ground_validation2,_,_,_, attention_validation = metric_vali_2[i].get_metrics( epoch+1) 
            total_loss_validation += loss_validation2
            logging.info(label + " validation loss at epoch " + str(epoch+1) + " -> " + str(loss_validation2))
            logging.info(label + " validation accuracy at epoch " + str(epoch+1) + " -> " + str(acc_validation2))
        logging.info("Total validation loss at epoch " + str(epoch+1) + " -> " + str(total_loss_validation))   

        ###################### TEST ALL VIEWS ###################
        metric_test_2 = train(
                test_loader2,
                model,
                criterion,
                metric_calculator,
                optimizer,
                epoch + 1,
                train=False,
            )

        # LOG INFORMATION ABOUT TEST
        total_loss_test = 0
        for i in range(len(metric_test_2)):
            if i == 0:
                label = 'offence_severity_class'
            elif i == 1:
                label = 'action_class'

            loss_test2, acc_test2, acc2_test2, classes_test2, classes2_test2, classes_ground_test2, total_predictions2, total_groundTruth2, classes_names2, attention_test =  metric_test_2[i].get_metrics(epoch+1) 
            total_loss_test += loss_test2
            logging.info(label + " test loss at epoch " + str(epoch+1) + " -> " + str(loss_test2))
            logging.info(label + " test accuracy at epoch " + str(epoch+1) + " -> " + str(acc_test2))
        logging.info("Total test loss at epoch " + str(epoch+1) + " -> " + str(total_loss_test))  
        
        if lr_warmup != 0:
            scheduler_warmup.step()
            lr_warmup -= 1
            if lr_warmup == 0:
                optimizer.param_groups[0]['lr'] = lr
        else:
            # Learning rate scheduler update
            prevLR = optimizer.param_groups[0]['lr']
            currLR = optimizer.param_groups[0]['lr']
            if scheduler_step == 1:
                scheduler.step(loss_validation2)
            elif scheduler_step == 0:
                pass
            elif scheduler_step == 4:
                if best_loss < loss_validation2:
                    gamma = gamma_aux
                else:
                    gamma = 1
                scheduler.step()
            else: 
                scheduler.step()


        """
        if total_loss_validation < best_loss:
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'vali_loss': loss_validation2,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch+1) + "_model.pth.tar")
            torch.save(state, path_aux)
        """

        # Remember best loss and save checkpoint
        best_loss = min(loss_validation2, best_loss)
        save_tests.append(acc_test2)

        # new best loss
        if best_loss == loss_validation2:
            counter = 0
            best_epoch = epoch
        else: 
            counter += 1

        if counter > 4:
                print("Reached plateau")
                return

def train(dataloader,
          model,
          criterion,
          metric_calculator, 
          optimizer,
          epoch,
          train=False,
        ):
    
    metric_calculator_offence_severity = metric_calculator[0]
    metric_calculator_action = metric_calculator[1]
    metric_calculator_offence_severity.reset()
    metric_calculator_action.reset()


    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    if True:
        for targets_offence_severity, targets_action, mvclips in dataloader:

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            
            mvclips = mvclips.cuda().float()
            #mvclips = mvclips.float()

            # compute output
            start=time.time()
            outputs_offence_severity, outputs_action, attention = model(mvclips)


            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            #compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = loss.item()
            metric_calculator_offence_severity.update(loss_offence_severity, outputs_offence_severity.detach().cpu(), targets_offence_severity.detach().cpu(), attention.detach().cpu())                        
            metric_calculator_action.update(loss_action, outputs_action.detach().cpu(), targets_action.detach().cpu(), attention.detach().cpu())

        gc.collect()
        torch.cuda.empty_cache()
    
    metric_calculator = [metric_calculator_offence_severity, metric_calculator_action]
    return metric_calculator