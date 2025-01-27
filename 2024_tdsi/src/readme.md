projet tdsi
Le répertoire est divise en plusieurs partie: 
    -répertoire data/: contient les images de la base de données STL-10 de test

    -répertoire data_model_training/: contient les images de la base de données STL-10 d'entrainement

    -répertoire model/: contient tous les modèles que nous avons pré-entrainé
        -pinv-net_70_lfcorr_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth : modele monomode entrainé sur 70_lfcorr avec régularisation des poids sur30 epochs

        -pinv-net_BF_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth modele monomode entrainé sur de la BF avec régularisation des poids sur 50 epochs

        -pinv-net_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth: modele monomode entrainé sur 70_lf avec régularisation des poids sur 50 epochs 

        -pinv-net_variance_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth: modele monomode entrainé sur variance avec régularisation des poids sur 30 epochs

        -right_noise_level_pinv-net_Unet_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth: modele monomode entrainé sur 70_lf SANS régularisation des poids sur 30 epochs
    -répertoire stat/: Contient les matrices de covariance nécessaire

    -répertoire src/:Contient tous les codes utiles
        -main.py
    répertoire misc/: Contient les scripts de définition de fonction, et de téléchargement des images nécessaire au fonctionnement 
        -download_data.py: Télécharge les données usuelles pour le fonctionnement de la libraire spyrit A EXECUTER AVANT DE TENTER LES AUTRES FICHIERS

        -download_images_stl10.py: Télécharge les images de la base de données STL-10 pour le test A EXECUTER AVANT DE TENTER LES AUTRES FICHIERS

        -pattern_order.py: Définit la fonction qui permet de définir les différents ordres d'acquisitions Variance, low_freq, 70_lf, 70_lfcorr,high_freq

        -Weight_Decay_Loss.py: Définit la loss qu'il faut pour la fonction de cout avec régularisation
        

explanation of the different file's role :
- download_data.py : download some natural images for different testing

- different_pattern_order_testing : reconstruction using different pattern order without denoising, and calculation of metrics PSNR and SSIM

- main.py : just a file for different purpose testing
- pattern_order.py : a function that sets the reconstruction order with img_size/4 measurements
- testing_trained_models.py : a file for testing different trained models  
- training_for_denoising.py : a file to train denoising models
-test_model_on_data.py:This file tests the performance of trained denoising models on specified datasets.
    -contains test_model_on_data(model_name=None,model_type=nnet.Unet, pattern_order=None,alpha=10,und=4,img_size=64,verbose=False,model_path=None)
    -contains use cases of test_model_on_data function
