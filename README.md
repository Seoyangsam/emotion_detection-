# emotion_detection-

Understanding children’s emotions associated with their food preferences can help us better understand their consumer behavior toward food products. The widespread adoption and usage of artificial intelligence specifically deep learning can be used to understand the emotions of children. Deep learning has been used for emotion detection in prior studies; however, the current datasets used for training the models are mainly focused on adults whereas the faces of children are not well represented and studied. The Child Affective Facial Expression (CAFE) dataset is a small- scale dataset that contains 7 different emotions of children from different backgrounds. Training deep learning models on this specific CAFE dataset for emotion detection is challenging due to the limited number of labeled data.<br>

Generative adversarial networks (GANs) are a type of deep learning model that consists of a generator and a discriminator. The goal of GANs is to generate synthetic data based on the training dataset. Progressive GAN (ProGAN) is a specific variant of GANs that generate detailed images using a progressive training approach. To tackle the small sample problem, we applied ProGAN for generating new images based on the CAFE dataset.<br>

We first trained our convolutional neural network model only on the original CAFE dataset. Later we generated new images via generative adversarial networks and retrained the model based on the new dataset, a mix of CAFE dataset and generated images. Our result shows that after using generative adversarial networks to increase the sample size, the convolutional neural network model's accuracy has improved from 0.93 to 0.96.<br>

The positive outcome proved that including generative adversarial networks can aid small sample problems in machine learning such as detecting the emotions of children. The outcome of this study will be used to detect children's emotions in video recordings of eating episodes afterward.<br>

# Requirements
Python 3.6 or higher<br>
TensorFlow 2.3.0<br>
Numpy<br>
Matplotlib<br>
OpenCV<br>
PyTorch<br> 

# Usage
1. Clone the repository: git clone https://github.com/Seoyangsam/emotion-detection-.git
2. Install the required packages: pip install -r requirements.txt
3. Download the CAFE dataset from the official website and place it in the project directory.
4. Data preprocessing: python split.py
5. Train the model: python CNN.py

# Additional Info
The ProGAN model used in this project is based on [this repo.](https://github.com/aladdinpersson/Machine-Learning-
Collection/blob/master/ML/Pytorch/GANs/ProGAN)<br>
The training process of ResNet implemented in JupyterHub of Wageningen University is about 10 days using CPU. 
The trained weights are saved in the saved_model directory, which is created after the training process.<br>
The test process will take the image path as an input and it will return the predicted emotion on that image.

# Contributing
This project is open for contributions. If you have any suggestions or bug reports, please open an issue or submit a pull request.

# References
[Child Affective Facial Expression dataset](https://nyu.databrary.org/volume/30)<br>
Washington, P.Y., Kalantarian, H., Kent, J., Husic, A., Kline, A., Leblanc, É., Hou, C., Mutlu, C., Dunlap, K., Penev, Y., Varma, M., Stockham, N.T., Chrisman, B.S., Paskov, K.M., Sun, M.W., Jung, J., Voss, C., Haber, N., & Wall, D.P. (2020). Training an Emotion Detection Classifier using Frames from a Mobile Therapeutic Game for Children with Developmental Disorders. ArXiv, abs/2012.08678. 
