# -*- coding: utf-8 -*-
"""
Copyright   I3S CNRS UCA 

This is an implementation of : Non-invasive live cell cycle monitoring using a supervised autoencoder
Written by : Philippe Pognonec, Axel Gustovic, Zied Djabari, Thierry Pourcher and Michel Barlaud 

Options : 
Select seed : line 43
Select train dataset : line 70 
Select test dataset : line 71 
Select number of control cells in the training set : line 74
Select number of cells in the test set : line 75

"""
import os
import sys
if '../functions/' not in sys.path:
    sys.path.append('../functions/')

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch import nn




# lib in '../functions/'
import functions.functions_torch as ft
import functions.functions_network_pytorch as fnp





#################################
  
if __name__=='__main__':
#------------ Parameters ---------
    # Set seed
    Seed = 5
    SEED = Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Set device (Gpu or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    nfold = 4
    N_EPOCHS = 10
    N_EPOCHS_MASKGRAD = 10      # number of epochs for trainning masked graident
    # learning rate 
    LR = 0.0005      
    BATCH_SIZE=8
    LOSS_LAMBDA = 0.0005         # Total loss =λ * loss_autoencoder +  loss_classification

    # Loss functions for reconstruction
    criterion_reconstruction = nn.SmoothL1Loss(  reduction='sum'  ) # SmoothL1Loss
    
    # Loss functions for classification
    criterion_classification = nn.CrossEntropyLoss( reduction='sum'   )
    


    TIRO_FORMAT = True

    file_name = '20210203_L1_All.csv'
    file_name2 = '20210203_M1_M2_M2_3_Mitosis.csv'
    
    
    Nc = 1000
    Nt = 5000
    
    
    # Choose Net 
#    net_name = 'LeNet'
    net_name = 'netBio'
    n_hidden = 64  # nombre de neurones sur la couche du netBio

    # Save Results or not
    SAVE_FILE = True
    # Output Path 
    outputPath =  'results/'+ 'CONTROL' + '/'
    if not os.path.exists(outputPath): # make the directory if it does not exist
        os.makedirs(outputPath)
        
    # Do pca or t-SNE 
    Do_pca = True
    Do_tSNE = True
    run_model= 'No_proj' 
    # Do projection at the middle layer or not
    DO_PROJ_middle = False
    # Wgether apply the ModelAnalyzer to the model
    PERFORM_MA = False
      
    # Do projection (True)  or not (False)
#    GRADIENT_MASK = False
    GRADIENT_MASK = True
    if GRADIENT_MASK:
        
        run_model='ProjectionLastEpoch'
    # Choose projection function
    if not GRADIENT_MASK:
        TYPE_PROJ = 'No_proj'
        TYPE_PROJ_NAME = 'No_proj'
    else:
#        TYPE_PROJ = ft.proj_l1ball         # projection l1
        TYPE_PROJ = ft.proj_l11ball        #original projection l11 (les colonnes a zero)
        TYPE_PROJ_NAME = TYPE_PROJ.__name__
        
    #  Parameters for gradient masqué  
    ETA = 1000         # for Proximal_PGL1 or Proximal_PGL11


 #   DoTopGenes = True
    DoTopGenes = False
          
#------------ Main loop ---------

    if Nc + Nt > pd.read_csv('./datas/' + str(file_name),delimiter=",", decimal=".",header=0).shape[0] :
        raise IndexError("Manque de données de control ")
    # Load data    
    ft.split_db(Nc, Nt, Seed, file_name, file_name2)
    X,Y,feature_name,label_name , X_test, Y_test = ft.ReadData(file_name ,'', file_name2, TIRO_FORMAT) # Load files datas

    feature_len = len(feature_name)
    class_len = len(label_name)
    print('Number of feature: {}, Number of class: {}'.format(feature_len,class_len ))
                    
    train_dl, vide, train_len, vide  = ft.SpiltData(X,Y,BATCH_SIZE)
    
    
    test_dl, vide, test_len, vide  = ft.SpiltData(X_test,Y_test,1)
    X_name = X 
    X = X[:,1:]
    X_name_test = X_test 
    X_test = X_test[:,1:]
    print('Len of train set: {}, Len of test set:: {}'.format(train_len,test_len))  

    accuracy_train = np.zeros((nfold,class_len+1))
    accuracy_test = np.zeros((nfold,class_len+1))
    data_train = np.zeros((nfold,3))
    data_test = np.zeros((nfold,3))
   
    for i in range(nfold) : 
        print('----------- Début iteration ',i,'----------------')
        # Define the SEED to fix the initial parameters 
        SEED = Seed + i
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        
        # run AutoEncoder
        if net_name == 'LeNet':
                net = ft.LeNet_300_100(n_inputs=feature_len, n_outputs=class_len).to(device)        # LeNet  
        if net_name == 'netBio':
                net = ft.netBio(feature_len, class_len , n_hidden).to(device)       # netBio  
     
        weights_entry,spasity_w_entry = fnp.weights_and_sparsity(net)
        topGenesCol_entry = ft.selectf(net.state_dict()['encoder.0.weight'] , feature_name, outputPath)
        
        if GRADIENT_MASK:
            run_model='ProjectionLastEpoch'
    
        optimizer = torch.optim.Adam(net.parameters(), lr= LR )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150, gamma = 0.1)
        data_encoder, data_decoded, epoch_loss, best_test, net = ft.RunAutoEncoder(net, criterion_reconstruction, optimizer, lr_scheduler, train_dl, train_len, test_dl, test_len, N_EPOCHS, \
                    outputPath, SAVE_FILE,  DO_PROJ_middle, run_model, criterion_classification, LOSS_LAMBDA, feature_name, TYPE_PROJ, ETA )  
        labelpredict = data_encoder[:,:-1].max(1)[1].cpu().numpy()
        # Do masked gradient
        
        
        if GRADIENT_MASK:
            print("\n--------Running with masked gradient-----")
            print("-----------------------")
            zero_list = []
            tol = 1.0e-3
            for index,param in enumerate(list(net.parameters())):
                if index<len(list(net.parameters()))/2-2 and index%2==0:
                    ind_zero = torch.where(torch.abs(param)<tol)
                    zero_list.append(ind_zero)
            
            # Get initial network and set zeros      
            # Recall the SEED to get the initial parameters
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            
            # run AutoEncoder
            if net_name == 'LeNet':
                net = ft.LeNet_300_100(n_inputs=feature_len, n_outputs=class_len).to(device)        # LeNet  
            if net_name == 'netBio':
                net = ft.netBio(feature_len, class_len , n_hidden).to(device)       # FairNet 
            optimizer = torch.optim.Adam(net.parameters(), lr= LR)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150,gamma=0.1)
            
            for index,param in enumerate(list(net.parameters())):
                if index<len(list(net.parameters()))/2-2 and index%2==0:
                    param.data[zero_list[int(index/2)]] =0 
                    
            run_model = 'MaskGrad'
            data_encoder, data_decoded, epoch_loss, best_test, net = ft.RunAutoEncoder(net, criterion_reconstruction, optimizer, lr_scheduler, train_dl, train_len, test_dl, test_len, N_EPOCHS_MASKGRAD, \
                    outputPath,  SAVE_FILE,  zero_list, run_model, criterion_classification, LOSS_LAMBDA, feature_name, TYPE_PROJ, ETA )    
            print("\n--------Finised masked gradient-----")
            print("-----------------------")
        #np.save(file_name.split('.')[0]+'_Loss_'+str(run_model), epoch_loss)
        
        data_encoder = data_encoder.cpu().detach().numpy() 
        data_decoded =  data_decoded.cpu().detach().numpy() 
        
        data_encoder_test, data_decoded_test, class_train, class_test , topGenesCol, data_pred, erreur = ft.runBestNet(train_dl, test_dl, best_test, outputPath , i, class_len, net, X_name_test, train_len, test_len, Nc, feature_name, run_model, X_name, file_name)
        accuracy_train[i] = class_train
        accuracy_test[i] = class_test
        # silhouette score
        X_encoder = data_encoder[:,:-1]
        labels_encoder = data_encoder[:,-1]
        data_encoder_test = data_encoder_test.cpu().detach().numpy()
        data_pred = data_pred.cpu().detach().numpy()


    

    rf = pd.read_csv('{}Cellules_rares.csv'.format(outputPath),delimiter=",", decimal=".",header=0  , index_col=0)
    rf = rf.where(rf!=6 , 'T')
    
    rf = rf.where(rf!=7 , 'Z')
    #rf.drop(columns = feature_name , inplace = True )
    rf.to_csv('{}Result_autoencoder_S{}_Nc{}.csv'.format(outputPath , SEED , Nc), sep = ',' , decimal ='.')
    

    ft.showCellResult(rf, nfold, label_name)

    df_accTrain, df_acctest = ft.showClassResult(accuracy_train, accuracy_test, nfold, label_name)
    # print sparsity  
    print('\n best test accuracy:',best_test/float(test_len))
    
    # Reconstruction by using the centers in laten space and datas after interpellation
    center_mean,  center_distance = ft.Reconstruction(0.2, data_encoder, net, class_len )
              
    # Do pca,tSNE for encoder data
    if Do_pca and Do_tSNE:
        #ft.ShowPcaTsne(X, Y, data_encoder, center_distance, class_len )
        ft.ShowPcaTsne(X, Y, data_pred, center_distance, class_len )

    
    # Do Implementation of Metropolis and pass to decoder for reconstruction
    #Metropolis_sampled = ft.DoMetropolis(net, data_encoder, Metropolis_len, class_len, feature_name, outputPath)
    
    
    

    # Get Top Genes of each class 

#    method = 'Shap'       # (SHapley Additive exPlanation) A nb_samples should be define
    nb_samples =300        # Randomly choose nb_samples to calculate their Shap Value, time vs nb_samples seems exponential 
    method = 'Captum_ig'   # Integrated Gradients
#    method = 'Captum_dl'  # Deeplift
#    method = 'Captum_gs'  # GradientShap
    if DoTopGenes:
        print("Running topGenes..." )
        df_topGenes = ft.topGenes(X.astype(float),Y,feature_name,class_len, feature_len, method, nb_samples, device, net)
        print("topGenes finished" )
        
        

    # Loss figure
    if os.path.exists(file_name.split('.')[0]+'_Loss_No_proj.npy') and os.path.exists(file_name.split('.')[0]+'_Loss_MaskGrad.npy'):
        loss_no_proj = np.load(file_name.split('.')[0]+'_Loss_No_proj.npy')
        loss_with_proj = np.load(file_name.split('.')[0]+'_Loss_MaskGrad.npy')
        plt.figure()
        plt.title(file_name.split('.')[0]+' Loss')
        plt.xlabel('Epoch')
        plt.ylabel('TotalLoss')
        plt.plot(loss_no_proj, label = 'No projection')
        plt.plot(loss_with_proj, label = 'With projection ')
        plt.legend()
        plt.show()
    if SAVE_FILE:
        df_acctest.to_csv('{}{}_acctest.csv'.format(outputPath,str(TYPE_PROJ_NAME)),sep=';') 
#        df_topGenes.to_csv('{}{}_topGenes_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';')
        print("Save topGenes results to: ' {} ' ".format(outputPath) )
        