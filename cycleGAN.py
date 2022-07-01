import tensorflow as tf
from tensorflow import keras
from keras import Model
import tensorflow_addons as tfa
import os
import random
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image
from google.colab import drive

drive.mount("/content/gdrive")
# Let's take a look at the features of some images in our dataset.
image = Image.open("/content/gdrive/MyDrive/Colab Notebooks/monet_jpg/000c1e3bff.jpg")

# Prints out some basic information about the image.
print("Format: {a}, Mode: {b}, Size: {c}".format(a = str(image.format), b = str(image.mode), c = str(image.size)))


# Create a function to help us load images into a numpy array.
def load_images(folder):
    loaded_images = []

    # Loop through the filename given, convert images to a numpy array, and append it to loaded images.
    for filename in os.listdir(folder):
        image_data = np.array(Image.open(folder + "/" + filename))
        loaded_images.append(image_data)

    return loaded_images

# Load in our data as numpy arrays.
df_monet = load_images('/content/gdrive/MyDrive/Colab Notebooks/monet_jpg')
df_photos = load_images('/content/gdrive/MyDrive/Colab Notebooks/photo_jpg')

# Let's take a look at some images we have
# The left column houses our Monet paintings
# The right column houses our background photos
for i in range(3):
    fig, ax = plt.subplots(1, 2)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(df_monet[i])
    ax[1].imshow(df_photos[i])


def discriminator_model(image_size):
    # Initialise weights.
    init = keras.initializers.RandomNormal(stddev=0.02)

    # Create input.
    input_img = keras.layers.Input(shape=image_size)

    # Begin passing the tensor throughout the layers.
    layer = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(input_img)
    # We will adjust the alpha (the parameter controlling the steepness for values less than zero) to the standard for Tensorflow, 0.3.
    layer = keras.layers.LeakyReLU(alpha=0.3)(layer)
    # We will sprinkle in some dropout layers to prevent overfitting.
    layer = keras.layers.Dropout(0.2)(layer)

    layer = keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.LeakyReLU(alpha=0.3)(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    layer = keras.layers.Conv2D(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.LeakyReLU(alpha=0.3)(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    # Generate the output.
    output_decision = keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(layer)

    # Build and compile model
    model = Model(input_img, output_decision)
    model.compile(
        loss='mse',
        optimizer='adam',
        loss_weights=[0.05]
    )

    return model


def gen_resnet(input_layer, n_filter):
    # Create the 2 CNN layers
    layer_1 = keras.layers.Conv2D(n_filter, (2, 2), padding='same')(input_layer)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer_1)
    layer_2 = keras.layers.Conv2D(n_filter, (2, 2), padding='same')(layer_1)

    # Merge the final layer with the input to create the shortcut connection.
    merged_layer = keras.layers.Concatenate()([layer_1, input_layer])

    return merged_layer


def generator_model(image_size, NO_RESNET):
    # Initialise weights.
    init = keras.initializers.RandomNormal(stddev=0.02)

    # Create input
    input_img = keras.layers.Input(shape=image_size)

    # Downsampling encoder.
    layer = keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=init)(input_img)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    # Decrease filter size for more specific feature mapping.
    layer = keras.layers.Conv2D(64, (2, 2), padding='same', kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.Dropout(0.5)(layer)

    # Begin implementing residual networks.
    for i in range(NO_RESNET):
        layer = gen_resnet(layer, 128)

    # Upsampling decoder.
    layer = keras.layers.Conv2DTranspose(64, (3, 3), padding='same', kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.Dropout(0.5)(layer)

    layer = keras.layers.Conv2DTranspose(32, (3, 3), padding='same', kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    layer = keras.layers.Conv2D(3, (3, 3), padding='same', kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=3)(layer)
    layer = keras.layers.Activation('relu')(layer)

    # Final output layer.
    output_img = keras.layers.Activation('tanh')(layer)

    # Build model.
    model = Model(input_img, output_img)

    return model


def combined_model(gen_1, gen_2, dis_1, image_size):
    # Isolate our model's training.
    gen_1.trainable = True
    dis_1.trainable = False
    gen_2.trainable = False

    input_img = keras.layers.Input(shape=image_size)

    #  Adversial loss
    gen_1_fake = gen_1(input_img)
    dis_1_output = dis_1(gen_1_fake)

    # Identiy loss
    input_img_2 = keras.layers.Input(shape=image_size)
    gen_1_output = gen_1(input_img_2)

    # Cycle loss
    # Forward cycle
    forward_output = gen_2(gen_1_fake)

    # Backward cycle
    gen_2_real = gen_2(input_img)
    backward_output = gen_1(gen_2_real)

    # Define model
    model = Model([input_img, input_img_2], [dis_1_output, gen_1_output, forward_output, backward_output])

    model.compile(
        loss=['mse', 'mae', 'mae', 'mae'],
        loss_weights=[1, 5, 10, 10],
        optimizer='adam'
    )

    return model


def save_model(step, gen_1, gen_2):
    step += 1

    gen_1.save('/Gen1/Generator 1 {a}.h5'.format(a=step))
    gen_2.save('/Gen2/Generator 2 {a}.h5'.format(a=step))

    print('Generator 1 and generator 2 saved for {a} step.'.format(a=step))

def update_img_pool(img_pool, images):

    selection = []

    for img in images:
        # If poolsize is already 50, we have to replace some images while using the replaced images.
        if len(img_pool) >= 50:
            index = random.randint(0, len(img_pool) - 1)
            selection.append(img_pool[index])
            img_pool[index] = img
        # To introduce some randomness: Sometimes we will use a image to train the model but not add it into the pool.
        elif random.uniform(0, 1) < 0.4:
            selection.append(img)
        # If there is space in the pool, we will just stock up.
        elif len(img_pool) < 50:
            img_pool.append(img)
            selection.append(img)

    # We should return the list as a array.
    selection = np.array(selection)
    return selection


def scaler(df):
    # Remember in the first cell of code that we are using images with size 256.
    factor = 256/2
    df = (df - factor)/factor
    return df


df_monet = scaler(df_monet)
df_photos = scaler(df_photos)


# Create a batch of real images
def generate_real(df, n_sample, output_size):
    index = np.random.randint(0, df.shape[0], n_sample)
    x = df[index]
    y = np.ones((n_sample, output_size, output_size, 1))

    return x, y


# Create a batch of fake images, we can pass in the output of generate_real to df
def generate_fake(df, gen, output_size):
    x = gen.predict(df)
    y = np.zeros((len(x), output_size, output_size, 1))

    return x, y


def performance(model_type, model, x, batch_size, step, output_size):
    # Create a batch of real images
    real_sample, placeholder = generate_real(x, batch_size, output_size)

    # Create a batch of fake images
    fake_sample, placeholder = generate_fake(x, model, output_size)

    # Scale the samples to all hold positive values
    real_sample = (real_sample + 1) / 2
    fake_sample = (fake_sample + 1) / 2

    # Save the images
    plt.axis('off')
    plt.imshow(fake_sample[5])

    name = "/outputimage/{a} plot step {b}.png".format(a=model_type, b=step)

    # Save our plot
    plt.savefig(name)
    plt.close()


def train_models(gen_1, gen_2, dis_1, dis_2, comb_1, comb_2, monet, photos):
    # Define some key variables
    EPOCHS = 10
    N_BATCH = 1
    OUTPUT_SIZE = dis_1.output_shape[1]

    # Develop our image pools
    pool_monet = []
    pool_photos = []

    # Find no. of training iterations we need
    batch_per_epoch = int(len(monet) / N_BATCH)
    STEPS = batch_per_epoch * EPOCHS

    for i in range(STEPS):

        # Generate real samples
        monet_real_x, monet_real_y = generate_real(monet, N_BATCH, OUTPUT_SIZE)
        photo_real_x, photo_real_y = generate_real(photos, N_BATCH, OUTPUT_SIZE)

        # Pass real samples into models to develop fakes
        monet_fake_x, monet_fake_y = generate_fake(photo_real_x, gen_1, OUTPUT_SIZE)
        photo_fake_x, photo_fake_y = generate_fake(monet_real_x, gen_2, OUTPUT_SIZE)

        # Add fakes into our pools
        monet_fake_x = update_img_pool(pool_monet, monet_fake_x)
        photo_fake_x = update_img_pool(pool_photos, photo_fake_x)

        # Update Gen1 (Photo - Monet) with combined model.
        # We will mainly focus on cycle and adversial loss first.
        gen_1_loss, _, _, _, _ = comb_1.train_on_batch([photo_real_x, monet_real_x],
                                                       [monet_real_y, monet_real_x, photo_real_x, monet_real_x])

        # Next we will update Dis1 - that tries to compare Monet paintings, with both real and fake Monet paintings.
        dis_1_loss_1 = dis_1.train_on_batch(monet_real_x, monet_real_y)
        dis_1_loss_2 = dis_1.train_on_batch(monet_fake_x, monet_fake_y)

        # Update Gen2 (Monet - Photo) with combined model again.
        # Once again, focus is placed on cycle and adversial loss.
        gen_2_loss, _, _, _, _ = comb_2.train_on_batch([monet_real_x, photo_real_x],
                                                       [photo_real_y, photo_real_x, monet_real_x, photo_real_x])
        # Update Dis2 - that tries to compare real photos.
        dis_2_loss_1 = dis_2.train_on_batch(photo_real_x, photo_real_y)
        dis_2_loss_2 = dis_2.train_on_batch(photo_fake_x, photo_fake_y)

        # Review performance
        print('Step: {a}, Dis1 loss: [{b}, {c}], Dis2 loss: [{d}, {e}], Gen1 loss: {f}, Gen2 loss: {g}'.format(
            a=i + 1, b=dis_1_loss_1, c=dis_1_loss_2, d=dis_2_loss_1, e=dis_2_loss_2, f=gen_1_loss, g=gen_2_loss
        ))

        e = i + 1
        # Let's evaluate our performance every 30 steps!
        if e % 30 == 0:
            # Gen1
            performance("Gen1", gen_1, photos, 5, i, OUTPUT_SIZE)
            # Gen2
            performance("Gen2", gen_2, monet, 5, i, OUTPUT_SIZE)

        # We will also save our model every 50 steps!
        elif e % 50 == 0:
            save_model(i, gen_1, gen_2)


# We will first extract the size of a single image, 256x256.
IMAGE_SIZE = df_monet.shape[1:]

# Start defining our models - let's try 5 ResNets for now
Gen1 = generator_model(IMAGE_SIZE, 5)
Gen2 = generator_model(IMAGE_SIZE, 5)

Dis1 = discriminator_model(IMAGE_SIZE)
Dis2 = discriminator_model(IMAGE_SIZE)

Photo_to_Monet = combined_model(Gen1, Gen2, Dis1, IMAGE_SIZE)
Monet_to_Photo = combined_model(Gen2, Gen1, Dis2, IMAGE_SIZE)