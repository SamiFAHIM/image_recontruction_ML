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
        -single_pixel.ipynb: le jupyter notebook qui permet de tester le différentes fonctions à savoir 
            -visualiser des ordres de reconstructions
            -faire des reconstructions par pseudo inverse
            -tester les différents modèles 
            -tester les différents modèles sur une base de tests
        -test_model_on_data.py: test les débruiteurs sur plusieurs données et sur plusieurs ordres d'inférence. Devrait pouvoir reconstruire la majorité des courbes que nous avons dans le rapport
        -training_for_denoising.py: permet de réaliser un entraînement pour les réseaux monomodes
        -training_Multiple_Acquisition.py: permet de réaliser un entrainement sur les réseaux multimodes

    répertoire misc/: Contient les scripts de définition de fonction, et de téléchargement des images nécessaire au fonctionnement 
        -download_data.py: Télécharge les données usuelles pour le fonctionnement de la libraire spyrit A EXECUTER AVANT DE TENTER LES AUTRES FICHIERS

        -download_images_stl10.py: Télécharge les images de la base de données STL-10 pour le test A EXECUTER AVANT DE TENTER LES AUTRES FICHIERS

        -pattern_order.py: Définit la fonction qui permet de définir les différents ordres d'acquisitions Variance, low_freq, 70_lf, 70_lfcorr,high_freq

        -Weight_Decay_Loss.py: Définit la loss qu'il faut pour la fonction de cout avec régularisation
    
Pour une execution correcte merci d'executer les scripts misc/download_data.py et misc/download_images_stl10.py

Si la base de données d'entrainement  de STL-10 n'est pas télécharger: il faut changer la ligne 106 de src/training_for_denoising.py ou bien de src/Training_Multiple_Acquisition_Matrix à 'True' afin de télécharger les données pour la première fois

Bonne Lecture et en cas de Question n'hesitez pas à nous contacter ;)