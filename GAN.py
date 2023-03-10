import glob
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape, Conv2DTranspose
import os
real_image_dataset = []

# Define the path to your data directory
data_dir = "./data"

# Loop through the subfolders (angry, disgust, fearful, happy, neutral, sad, surprise)
for emotion_folder in ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprise"]:
    # Build the path to the current subfolder
    current_folder = os.path.join(data_dir, emotion_folder)

    # Loop through the images in the current subfolder
    for filename in os.listdir(current_folder):
        # Build the path to the current image
        filepath = os.path.join(current_folder, filename)

        # Add the image path to the real_image_dataset list
        real_image_dataset.append(filepath)

print(real_image_dataset)
print(type(real_image_dataset))


# assuming real_image_dataset is a list of image file paths
images = []
for image_path in real_image_dataset:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    images.append(image)

real_image_dataset = np.array(images)
real_image_dataset = (real_image_dataset - 0.5) / 0.5

# discriminator model
def generate_discriminator(in_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


# generator model
def generate_generator(latent_dim):
    model = Sequential()

    n_nodes = 256 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 256)))

    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model
# GAN model
def generate_GAN(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model



# select real samples
def generate_real_samples(dataset, n_samples):
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    y = np.full(n_samples, 0.85)

    return X, y

# generate fake samples
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.full(n_samples, 0.1)

    return X, y

def generate_latent_points(latent_dim, n_samples) :
    x_input = np.random.rand(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input


# training
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    batches_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        for j in range(batches_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            discriminator.trainable = True
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            discriminator.trainable = False
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, batches_per_epoch, d_loss1, d_loss2, g_loss))

latent_dim = 100
discriminator = generate_discriminator()
generator = generate_generator(latent_dim)
gan_model = generate_GAN(generator, discriminator)
train(generator, discriminator, gan_model, real_image_dataset, latent_dim)
