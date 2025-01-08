projet tdsi

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
