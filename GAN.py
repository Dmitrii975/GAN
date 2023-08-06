import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import cv2
from PIL import Image

cross_entropy = BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
descriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
EPOCHS = 50
BATCH_SIZE = 256
BUFFER_SIZE = 8000

def create_descriminator(input_shape):
  dropout = 0.3

  model = Sequential()

  model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(dropout))
  assert model.output_shape == (None, 75, 75, 128)

  model.add(layers.Conv2D(128, (3,3), strides=(3,3), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(dropout))
  assert model.output_shape == (None, 25, 25, 128)

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

def create_generator():
  model = Sequential()

  model.add(layers.Dense(25*25*64, input_shape=(200,), use_bias=False))
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((25,25,64)))
  assert model.output_shape == (None, 25, 25, 64)

  model.add(layers.Conv2DTranspose(128, (5,5), padding='same', use_bias=False, strides=(3,3)))
  assert model.output_shape == (None, 75, 75, 128)
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5,5), padding='same', use_bias=False, strides=(2,2)))
  assert model.output_shape == (None, 150, 150, 64)
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5,5), padding='same', use_bias=False, strides=(1,1)))
  assert model.output_shape == (None, 150, 150, 1)

  return model

def generator_loss(fake_output):
  # Считаем, что нам должно прилететь решение дескриминатора 1 т.е np.ones_like(fake_output) ну и подгоняем генератор под это
  loss = cross_entropy(tf.ones_like(fake_output), fake_output)
  return loss

def descriminator_loss(real_output, fake_output):
  # Паралельно учим и дескриминатор. Мы скармливаем ему сначала норм фото, а потом загенерированные real_output и fake_output соответственно и учим его правильно определять все это дело
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

  # Считаем общие его потери, пускай сам все подгоняет
  total_loss = real_loss + fake_loss

  return total_loss

#Функция, прогоняющая 1 эпоху и получающая картинки для прогона
@tf.function
def train_epoch(images, descriminator, generator):

  noise_for_generator = np.random.rand(BATCH_SIZE,200)

  with tf.GradientTape() as descr_tape, tf.GradientTape() as gen_tape:
    # генерируем фото на основе шума
    generated_images = generator(noise_for_generator, training=True)

    # спрашиваем дескриминатор насчет реальных фото
    real_output = descriminator(images, training=True)
    # спрашиваем дескриминатор насчет сгенерированных фото
    fake_output = descriminator(generated_images, training=True)

    #Считаем потери
    generator_losses = generator_loss(fake_output)
    descriminator_losses = descriminator_loss(real_output, fake_output)

    #Считаем градиентный спуск
    gen_gradients = gen_tape.gradient(generator_losses, generator.trainable_variables)
    descr_gradients = descr_tape.gradient(descriminator_losses, descriminator.trainable_variables)

    # Применяем спуск к нейронке
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    descriminator_optimizer.apply_gradients(zip(descr_gradients, descriminator.trainable_variables))

def prepare_images(der):
  print("Preparing images...")
  images = []
  for i in os.listdir(der):
    image = cv2.imread(der + i)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(gray_image)
  numpy_images = np.array(images)[:8000] // 255
  numpy_images = np.reshape(numpy_images, (8000, 150, 150, 1))
  dataset = tf.data.Dataset.from_tensor_slices(numpy_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  return dataset

def show_result_after_epoch(generator: Sequential):
  noise = np.random.rand(200)
  noise = np.expand_dims(noise, 0)
  prediction = generator.predict(noise, verbose=False)
  plt.imshow(prediction[0, :, :, 0], cmap='gray')
  plt.show()

def save_result(generator: Sequential, epoch):
  noise = np.random.rand(200)
  noise = np.expand_dims(noise, 0)
  prediction = generator.predict(noise, verbose=False)
  prediction = prediction[0, :, :, :]
  #print(prediction.shape)
  prediction[prediction < 0] = 0
  #print(prediction)
  prediction = prediction * 255
  #print(prediction)
  image = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
  #print(image.shape)
  cv2.imwrite('/content/drive/MyDrive/results_generator/' + str(epoch + 1) + '.jpg', image)

def train(epochs):
  images = prepare_images('/content/drive/MyDrive/ready_photos/dogs_ready/')
  descriminator = create_descriminator((150,150,1))
  generator = create_generator()

  print("Starting training...")
  for i in range(epochs):
    print("Epoch number: " + str(i + 1))
    for j, batch in enumerate(images):
      print('   Batch number: ' + str(j + 1))
      train_epoch(batch, descriminator, generator)
    save_result(generator)
train(EPOCHS)


