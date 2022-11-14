"""
Contains the loss function, the model training code and the predictions from the model
"""

import torch
from torch import nn
import numpy as np
import math

def eucl_loss(output,target):
    loss = (output-target).pow(2).sum(2).sqrt().mean()
    return loss

def model_training(train_loader, train_ts, model, optimizer, list_edges, epochs):
    for i in range(epochs):
        y_preds = []
        for j,data in enumerate(train_loader):
            x=data[0]
            y=data[1]
            t=data[2]
            optimizer.zero_grad()
            
            input_list = [torch.tensor(x.float())[:,i,:,:] for i in range(0,22)]
            ts_list = [data[2].float()[:,i,:] for i in range(0,22)]
            y_pred = model(input_list,ts_list,torch.tensor(list_edges).t().contiguous())
            y_preds.append(y_pred)
            single_loss = eucl_loss(y_pred, y.float())
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    return model, y_preds

def get_test_predictions(test_loader, model, y_test, list_edges, scaler):
    test_preds = []
    model.eval()
    
    for j,data in enumerate(test_loader):
        x=data[0]
        y=data[1]
        t=data[2]
        input_list = [torch.tensor(x.float())[:,i,:,:] for i in range(0,22)]
        ts_list = [data[2].float()[:,i,:] for i in range(0,22)]
        with torch.no_grad():
            test_preds.append(model(input_list,ts_list,torch.tensor(list_edges).t().contiguous()))
    
    list_vals = [t.tolist() for t in test_preds]
    list_vals = np.array([i for sublist in list_vals for item in sublist for i in item]).reshape(int(y_test.shape[0]/22),22,2)
    actual_predictions = np.array([scaler.inverse_transform(ype) for ype in list_vals]).reshape(y_test.shape[0],2)
    y_test_example = y_test.reshape(int(y_test.shape[0]/22),22,2)
    actual_test_results = np.array([scaler.inverse_transform(yte) for yte in y_test_example])
    return actual_predictions, actual_test_results