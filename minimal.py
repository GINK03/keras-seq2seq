from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model


timesteps   = 10
input_dim   = 10
latent_dim  = 512 
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)
encoder = Model(inputs, encoded)

x = RepeatVector(timesteps)(encoded)
x = LSTM(input_dim, return_sequences=True)(x)
decoded = Dense(10, activation='softmax')(x)

autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

import numpy as np
import random

xss = []
for _ in range(1000):
  xs = []
  for _ in range(0,10):
    ba = [0.]*10
    ba[random.randint(0,9)] = 1
    xs.append( ba )
  xss.append( xs ) 

yss = []
for _ in range(1000):
  ys = []
  for _ in range(0,10):
    ba = [0.]*10
    ba[random.randint(0,9)] = 1
    ys.append( ba )
  yss.append( ys )

Xs = np.array( xss ) 
Ys = np.array( yss )

autoencoder.fit(Xs, Ys)



