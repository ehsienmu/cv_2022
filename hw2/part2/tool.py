


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt

from myDatasets import PsudeoDataset

def get_pseudo_labels(dataset, model, batch_size, threshold=0.99):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    # Iterate over the dataset by batches.
    new_targets = []
    relabel_indices = []
    for i, batch in enumerate(tqdm(data_loader, leave=False)):
        # img, labels = batch
        img = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        prob, labels = torch.max(probs, 1)  
        for sub_idx in range(prob.shape[0]):
            if prob[sub_idx] >= threshold:
                relabel_indices.append(i * batch_size + sub_idx)
                new_targets += [labels[sub_idx].item()]


    #     # ---------- TODO ----------
    #     # Filter the data and construct a new dataset

    subset = torch.utils.data.Subset(dataset, relabel_indices)
    newset = PsudeoDataset(subset, new_targets)

    print(f'{len(relabel_indices)} more data available')

    # # Turn off the eval mode.
    model.train()
    return newset


def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        
        
def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)#, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##
def plot_learning_curve(x, y, log_path, plot_title):
    """_summary_
    The function is mainly to show and save the learning curves. 
    input: 
        x: data of x axis 
        y: data of y axis 
    output: None 
    """
    #############
    ### TO DO ### 
    # You can consider the package "matplotlib.pyplot" in this part.
    
    
    plt.figure()
    plt.plot(x, y)
    plt.title(plot_title)
    plt.savefig(log_path + plot_title + '.png')
    # plt.savefig(f"./{plot_title}.png")
    # pass
    

def train(model, train_loader, val_loader, num_epoch, early_stop, log_path, save_path, device, criterion, scheduler, optimizer, batch_size, do_semi, train_set, unlabeled_set):
    start_train = time.time()
    # overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    # overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    # overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    # overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    overall_loss = [] #np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = [] #np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = [] #np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = [] #np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0.0
    last_val_acc = 0.0
    last_last_val_acc = 0.0
    early_stop_cnt = 0
    for i in range(num_epoch):
        print(f'epoch = {i}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0
        train_acc = 0.0

        if do_semi and (last_val_acc > 0.7 and last_val_acc >= last_last_val_acc):
            last_last_val_acc = last_val_acc
            # Obtain pseudo-labels for unlabeled data using trained model.
            print("Let get_pseudo_labels")
            pseudo_set = get_pseudo_labels(unlabeled_set, model, batch_size, threshold=.9)

            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.

            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)



        # training part
        # start training
        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            loss = criterion(output, label)
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        # scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader.dataset) 
        train_acc = corr_num / len(train_loader.dataset)
                
        # record the training loss/acc
        # overall_loss[i], overall_acc[i] = train_loss, train_acc
        overall_loss.append(train_loss)
        overall_acc.append(train_acc)

        #############
        ## TO DO ##
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            corr_num = 0
            val_acc = 0.0
            
            ## TO DO ## 
            # Finish forward part in validation. You can refer to the training part 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

            for batch_idx, (data, label,) in enumerate(tqdm(val_loader)):
                # put the data and label on the device
                # note size of data (B,C,H,W) --> B is the batch size
                data = data.to(device)
                label = label.to(device)

                # pass forward function define in the model and get output 
                output = model(data) 

                # calculate the loss between output and ground truth
                loss = criterion(output, label)
                
                # # discard the gradient left from former iteration 
                # optimizer.zero_grad()

                # # calcualte the gradient from the loss function 
                # loss.backward()
                
                # # if the gradient is too large, we dont adopt it
                # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
                
                # # Update the parameters according to the gradient we calculated
                # optimizer.step()

                val_loss += loss.item()

                # predict the label from the last layers' output. Choose index with the biggest probability 
                pred = output.argmax(dim=1)
                
                # correct if label == predict_label
                corr_num += (pred.eq(label.view_as(pred)).sum().item())

            # scheduler += 1 for adjusting learning rate later
            
            # averaging training_loss and calculate accuracy
            val_loss = val_loss / len(val_loader.dataset) 
            val_acc = corr_num / len(val_loader.dataset)
            last_val_acc = val_acc        
            # record the training loss/acc
            # overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc
            overall_val_loss.append(val_loss)
            overall_val_acc.append(val_acc)
            scheduler.step(val_loss)
            # scheduler.step()
        #####################
        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        #torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        early_stop_cnt += 1
        # save the best model if it gain performance on validation set
        if  val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            early_stop_cnt = 0
        if early_stop_cnt > early_stop:
            print('early stop!')
            break
        

    ## original version 
    # x = range(0, num_epoch)
    # overall_acc = overall_acc.tolist()
    # overall_loss = overall_loss.tolist()
    # overall_val_acc = overall_val_acc.tolist()
    # overall_val_loss = overall_val_loss.tolist()

    # list version
    x = range(0, len(overall_acc))

    # Plot Learning Curve
    ## TO DO ##
    # Consider the function plot_learning_curve(x, y) above
    
    # pass
    
    plot_learning_curve(x, overall_acc, save_path, 'Training_Accuracy')
    plot_learning_curve(x, overall_loss, save_path, 'Training_Loss')
    plot_learning_curve(x, overall_val_acc, save_path, 'Validation_Accuracy')
    plot_learning_curve(x, overall_val_loss, save_path, 'Validation_Loss')


