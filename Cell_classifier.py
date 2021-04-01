# -*- coding: utf-8 -*-
"""
Copyright   I3S CNRS UCA 

Select train dataset : line 197 
Select test dataset : line 198 
Select number of control cells in the training set : line 201
Select number of cells in the test set : line 202

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

from sklearn import metrics
from sklearn.model_selection import train_test_split


# lib in '../functions/'
import functions.functions_torch as ft
import functions.functions_network_pytorch as fnp
from sklearn.preprocessing import scale as scale





def split_db (NC, NT) : 
    """
    Read the data and complete the database of cells in mitosis with randomly selected control cells
    
    """
        # ==========Read the dataset===========
    df_X1 = pd.read_csv('./datas/' + str(file_name),delimiter=",", decimal=".",header=0)
    np.random.seed(SEED)    

    df_X2 = pd.read_csv('./datas/' + str(file_name2),delimiter=",", decimal=".",header=0)
        
    N1 = ['C'+str(i) for i in range(df_X1.shape[0])]
    df_X1.insert(0,"NAME", N1, True)
    
    M1 = ['M'+str(i) for i in range(df_X2.shape[0])]
    LabelM = [ 1 for i in range(df_X2.shape[0])]
    df_X2.insert(0,"NAME", M1, True)
    df_X2.insert(1,"LABEL", LabelM, True)
    

    # ==========Generate train and test dataset===========
   
    Nc=NC # number of samples for training 
    
    df_sample , df_TEST  = train_test_split(df_X1 , test_size= 1 - Nc/df_X1.shape[0] , random_state = SEED )
    df_TEST , no  = train_test_split(df_TEST , test_size= 1 - NT/df_TEST.shape[0] , random_state = SEED )
    df_TEST = df_TEST.sort_values(by='NAME')

    col_label =3*np.ones(len(df_sample))
    df_sample.insert(1,'LABEL',col_label)
    df_TRAIN=pd.concat([df_X2,df_sample],sort=False).reset_index(drop=True)
    T3 = df_TRAIN.where(df_TRAIN != 2 )
    df_TRAIN = pd.DataFrame(T3.dropna())

    col_labelt =3*np.ones(len(df_TEST))
    df_TEST.insert(1,'LABEL',col_labelt)
    
    
    df_TRAIN.T.to_csv('./datas/Train_cell.csv', decimal=".",header=0, sep = ',')
    df_TEST.T.to_csv('./datas/Test_cell.csv', decimal="." ,header = 0, sep = ',')
    
    
def ReadData(file_name , model, file_name2):
    """Read different data(csv, npy, mat) files  
    * csv has two format, one is data of facebook, another is TIRO format.
    
    Args:
        file_name: string - file name, default directory is "datas/FAIR/"
        
    Returns:
        X(m*n): numpy array - m samples and n features, normalized   
        Y(m*1): numpy array - label of all samples (represented by int number, class 0,class 1，2，...)
        feature_name(n*1): string -  name of features
        label_name(m*1): string -  name of each class
    """
    global TIRO_FORMAT
    if (file_name.split('.')[-1] =='csv'):
        if(model == 'autoencoder'):
            data_pd = pd.read_csv(str(file_name),delimiter=',', decimal=".", header=0, encoding = 'ISO-8859-1')
            X = (data_pd.iloc[1:,1:].values.astype(float)).T
            Y = data_pd.iloc[0,1:].values.astype(float).astype(int)
            feature_name = data_pd['Name'].values.astype(str)[1:]
            label_name = np.unique(Y)
        elif not TIRO_FORMAT:
            data_pd = pd.read_csv( str(file_name),delimiter=',',header=None,dtype='unicode')
            
            index_root = data_pd[data_pd.iloc[:,-1]=='root'].index.tolist()
            data = data_pd.drop(index_root).values
            X = data[1:,:-1].astype(float)
            Y = data[1:,-1]
            feature_name = data[0,:-1]
            label_name = np.unique(data[1:,-1])
            # Do standardization
            X = X-np.mean(X,axis=0)
            #X = scale(X,axis=0)    
        
        elif TIRO_FORMAT:
            data_pd = pd.read_csv( './datas/Train_cell.csv',delimiter=',', decimal=".", header=0, encoding = 'ISO-8859-1')
            Name = data_pd.columns
            data_pd.drop([1 , 20, 17, 18, 19], 0, inplace=True)
            X = (data_pd.iloc[1:,1:].values.astype(float)).T
            #X = np.vstack((Name[1:] , X))
            #X = X.T
            Y = data_pd.iloc[0,1:].values.astype(float).astype(int)
            feature_name = data_pd['NAME'].values.astype(str)[1:]
            label_name = np.unique(Y)
            
            data_pd_test = pd.read_csv( './datas/Test_cell.csv',delimiter=',', decimal=".", header=0, encoding = 'ISO-8859-1')
            
            Name_t = data_pd_test.columns
            data_pd_test.drop([1 , 20, 17, 18, 19], 0, inplace=True)
            X_test = (data_pd_test.iloc[1:,1:].values.astype(float)).T
            
            Y_test = data_pd_test.iloc[0,1:].values.astype(float).astype(np.int64)
            # Do standardization
            X  = np.log(abs(X)+1) 
            Xr = X        
            Y_c = np.where(Y == 3)
            
            X_c = Xr[Y_c]               # Transformation            
            
            X = X-np.mean(X_c,axis=0) 
            #X = X-np.mean(X,axis=0)                    
                       
            X_test = np.log(abs(X_test) +1)
            X_test = X_test - (np.mean(X_c, axis=0)) 
            #X_test = X_test - (np.mean(X_test, axis=0)) 
           
                
            X = np.vstack((Name[1:] , X.T)).T
            X_test = np.vstack((Name_t[1:] , X_test.T)).T
             
        for index,label in enumerate(label_name):   # convert string labels to numero (0,1,2....)
            Y = np.where(Y==label,index,Y)
        Y = Y.astype(np.int64) 
        for index,label in enumerate(label_name):   # convert string labels to numero (0,1,2....)
            Y_test = np.where(Y_test==label,index,Y_test)
        Y_test = Y_test.astype(np.int64)
        
    return X,Y,feature_name,label_name , X_test, Y_test



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
    
    nfold = 1
    N_EPOCHS = 1
    N_EPOCHS_MASKGRAD = 1     # number of epochs for trainning masked graident
    # learning rate 
    LR = 0.0005      
    BATCH_SIZE=8
    LOSS_LAMBDA = 0.0005         # Total loss =λ * loss_autoencoder +  loss_classification
    INTERPELLATION_LAMBDA = 0.2      # z = (1-λ)*x + λ*y.
    Metropolis_len = 100    # Number of points for Metropolis per class
    
    SPLIT_RATE = 0.25
    # Loss functions for reconstruction
#    criterion_reconstruction = nn.KLDivLoss( reduction='sum' )      # Kullback-Leibler Divergence loss function
#    criterion_reconstruction = CMDS_Loss(  reduction='sum'  )     # CMDS and Squared-output regularizer
#    criterion_reconstruction = nn.MSELoss(  reduction='sum'  ) # MSELoss
#    criterion_reconstruction = nn.L1Loss(  reduction='sum'  )  # L1Loss
    criterion_reconstruction = nn.SmoothL1Loss(  reduction='sum'  ) # SmoothL1Loss
    
    # Loss functions for classification
    criterion_classification = nn.CrossEntropyLoss( reduction='sum'   )
    


    TIRO_FORMAT = True

    file_name = '20210203_L1_All.csv'
    file_name2 = '20210203_M1_M2_M2_3_Mitosis.csv'
    
    
    Nc = 1000
    Nt = 3000
    
    
    # Choose Net 
#    net_name = 'LeNet'
#    net_name = 'Fair'
    net_name = 'netBio'
    n_hidden = 64  # nombre de neurone sur la couche du netBio

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
#        TYPE_PROJ = ft.proj_l11ball_line   #projection l11 (les lignesa zero)
#        TYPE_PROJ = ft.proj_nuclear         # projection Nuclear
#        TYPE_PROJ = ft.proj_l21ball        # projection l21
#        TYPE_PROJ = ft.proj_l12ball
        TYPE_PROJ_NAME = TYPE_PROJ.__name__
        
    #  Parameters for gradient masqué  
    ETA = 1000         # for Proximal_PGL1 or Proximal_PGL11
    ETA_STAR = 100.0   # for Proximal_PGNuclear or Proximal_PGL1_Nuclear
    AXIS = 0          #  for PGL21
  
    # Top genes params 

 #   DoTopGenes = True
    DoTopGenes = False
          
#------------ Main loop ---------

    if Nc + Nt > pd.read_csv('./datas/' + str(file_name),delimiter=",", decimal=".",header=0).shape[0] :
        raise IndexError("Manque de données de control ")
    # Load data    
    split_db(Nc, Nt)
    X,Y,feature_name,label_name , X_test, Y_test = ReadData(file_name ,'', file_name2) # Load files datas

    feature_len = len(feature_name)
    class_len = len(label_name)
    print('Number of feature: {}, Number of class: {}'.format(feature_len,class_len ))
                    
    train_dl, vide, train_len, vide  = ft.SpiltData(X,Y,BATCH_SIZE,SPLIT_RATE)
    
    
    test_dl, vide, test_len, vide  = ft.SpiltData(X_test,Y_test,1,SPLIT_RATE)
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
        if net_name == 'Fair':
                net = ft.FairAutoEncodert(feature_len, class_len ).to(device)       # FairNet  
        if net_name == 'netBio':
                net = ft.netBio(feature_len, class_len , n_hidden).to(device)       # netBio  
     
        weights_entry,spasity_w_entry = fnp.weights_and_sparsity(net)
        topGenesCol_entry = ft.selectf(net.state_dict()['encoder.0.weight'] , feature_name, outputPath)
        
        if GRADIENT_MASK:
            run_model='ProjectionLastEpoch'
    
        optimizer = torch.optim.Adam(net.parameters(), lr= LR )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150, gamma = 0.1)
        data_encoder, data_decoded, epoch_loss, best_test, net = ft.RunAutoEncoder(net, criterion_reconstruction, optimizer, lr_scheduler, train_dl, train_len, test_dl, test_len, N_EPOCHS, \
                    outputPath, SAVE_FILE,  DO_PROJ_middle, run_model, criterion_classification, LOSS_LAMBDA, feature_name, TYPE_PROJ, ETA, ETA_STAR, AXIS )  
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
            if net_name == 'Fair':
                net = ft.FairAutoEncodert(feature_len, class_len ).to(device)       # FairNet 
            if net_name == 'netBio':
                net = ft.netBio(feature_len, class_len , n_hidden).to(device)       # FairNet 
            optimizer = torch.optim.Adam(net.parameters(), lr= LR)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150,gamma=0.1)
            
            for index,param in enumerate(list(net.parameters())):
                if index<len(list(net.parameters()))/2-2 and index%2==0:
                    param.data[zero_list[int(index/2)]] =0 
                    
            run_model = 'MaskGrad'
            data_encoder, data_decoded, epoch_loss, best_test, net = ft.RunAutoEncoder(net, criterion_reconstruction, optimizer, lr_scheduler, train_dl, train_len, test_dl, test_len, N_EPOCHS_MASKGRAD, \
                    outputPath,  SAVE_FILE,  zero_list, run_model, criterion_classification, LOSS_LAMBDA, feature_name, TYPE_PROJ, ETA, ETA_STAR, AXIS )    
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
    
        data_train[i,0] = metrics.silhouette_score(X_encoder, labels_encoder, metric='euclidean')
        

        X_encodertest = data_encoder_test[:,:-1]
        labels_encodertest = data_encoder_test[:,-1]
        #data_test[i,0]  = metrics.silhouette_score(X_encodertest, labels_encodertest, metric='euclidean')        
        # ARI score
        data_train[i,1]  = metrics.adjusted_rand_score(labels_encoder, labelpredict)
        #data_test[i,1] = metrics.adjusted_rand_score(Y_test, data_encoder_test[:,:-1].max(1)[1].numpy())
        # AMI Score 
        data_train[i,2]  = metrics.adjusted_mutual_info_score(labels_encoder, labelpredict)
        #data_test[i,2] = metrics.adjusted_mutual_info_score(Y_test,data_encoder_test[:,:-1].max(1)[1].numpy() )

    

    rf = pd.read_csv('{}Cellules_rares.csv'.format(outputPath),delimiter=",", decimal=".",header=0  , index_col=0)
    rf = rf.where(rf!=6 , 'T')
    
    rf = rf.where(rf!=7 , 'Z')
    #rf.drop(columns = feature_name , inplace = True )
    rf.to_csv('{}Result_autoencoder_S{}_Nc{}.csv'.format(outputPath , SEED , Nc), sep = ',' , decimal ='.')
    

    ft.showCellResult(rf, nfold, label_name)

    df_accTrain, df_acctest = ft.showClassResult(accuracy_train, accuracy_test, nfold, label_name)
    df_metricsTrain, df_metricstest = ft.showMetricsResult(data_train, data_test, nfold)
    # print sparsity  
    print('\n best test accuracy:',best_test/float(test_len))
    
    # Reconstruction by using the centers in laten space and datas after interpellation
    center_mean,  center_distance = ft.Reconstruction(INTERPELLATION_LAMBDA, data_encoder, net, class_len )
              
    # Do pca,tSNE for encoder data
    if Do_pca and Do_tSNE:
        #ft.ShowPcaTsne(X, Y, data_encoder, center_distance, class_len )
        ft.ShowPcaTsne(X, Y, data_pred, center_distance, class_len )

    
    # Do Implementation of Metropolis and pass to decoder for reconstruction
    #Metropolis_sampled = ft.DoMetropolis(net, data_encoder, Metropolis_len, class_len, feature_name, outputPath)

    


    # get weights and spasity
    spasity_percentage_entry = {}
    for keys in spasity_w_entry.keys():
        spasity_percentage_entry[keys]= spasity_w_entry[keys]*100
    print('spasity % of all layers entry \n',spasity_percentage_entry)
    print("-----------------------")
    weights,spasity_w = fnp.weights_and_sparsity(net.encoder)
    spasity_percentage = {}
    for keys in spasity_w.keys():
        spasity_percentage[keys]= spasity_w[keys]*100
    print('spasity % of all layers \n',spasity_percentage)
    print("-----------------------")
    
    weights_decoder,spasity_w_decoder = fnp.weights_and_sparsity(net.decoder)
    mat_in =   net.state_dict()['encoder.0.weight']        

    mat_col_sparsity = ft.sparsity_col(mat_in, device = device)
    print(" Colonnes sparsity sur la matrice d'entrée: \n",mat_col_sparsity )
    mat_in_sparsity = ft.sparsity_line(mat_in, device = device)
    print(" ligne sparsity sur la matrice d'entrée: \n",mat_in_sparsity )
    layer_list = [x for x in weights.values() ]
    layer_list_decoder = [x for x in weights_decoder.values() ]
    titile_list = [x for x in spasity_w.keys()]
    ft.show_img(layer_list,layer_list_decoder, file_name2)
    
    
    

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
        