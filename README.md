# emotion_detection-

This project uses a convolutional neural network (CNN) to detect emotions from images of human faces. The model is trained on the Child Affective Facial Expression (CAFE) dataset, which contains images of children's faces labeled with one of seven emotions: angry, disgusted, fearful, happy, neutral, sad, and surprised.

# Requirements
Python 3.6 or higher<br>
TensorFlow 2.3.0<br>
Numpy<br>
Matplotlib<br>
OpenCV<br>

# Usage
1. Clone the repository: git clone https://github.com/Seoyangsam/emotion-detection-cnn.git
2. Install the required packages: pip install -r requirements.txt
3. Download the CAFE dataset from the official website and place it in the project directory.
4. Train the model: python train.py
5. Test the model: python test.py

# Additional Info
The model architecture used in this project is based on [this paper.](https://pediatrics.jmir.org/2022/2/e26760)<br>
The training process takes around 5-10 minutes on a GPU and 20-30 minutes on a CPU.<br>
The trained weights are saved in the saved_model directory, which is created after the training process.<br>
The test process will take the image path as an input and it will return the predicted emotion on that image.

# Contributing
This project is open for contributions. If you have any suggestions or bug reports, please open an issue or submit a pull request.

# References
[Child Affective Facial Expression dataset](https://nyu.databrary.org/volume/30)<br>
Washington, P.Y., Kalantarian, H., Kent, J., Husic, A., Kline, A., Leblanc, É., Hou, C., Mutlu, C., Dunlap, K., Penev, Y., Varma, M., Stockham, N.T., Chrisman, B.S., Paskov, K.M., Sun, M.W., Jung, J., Voss, C., Haber, N., & Wall, D.P. (2020). Training an Emotion Detection Classifier using Frames from a Mobile Therapeutic Game for Children with Developmental Disorders. ArXiv, abs/2012.08678. 
