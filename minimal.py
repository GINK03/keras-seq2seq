from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
import numpy as np
import random
import sys

timesteps   = 50
input_dim   = 128
latent_dim  = 512 
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)
encoder = Model(inputs, encoded)

x = RepeatVector(timesteps)(encoded)
x = LSTM(input_dim, return_sequences=True)(x)
decoded = Dense(128, activation='softmax')(x)

autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')


def test():
  xss = []
  for _ in range(1000):
    xs = []
    for _ in range(0,10):
      ba = [0.]*128
      ba[random.randint(0,9)] = 1
      xs.append( ba )
    xss.append( xs ) 

  yss = []
  for _ in range(1000):
    ys = []
    for _ in range(0,10):
      ba = [0.]*128
      ba[random.randint(0,9)] = 1
      ys.append( ba )
    yss.append( ys )
  Xs = np.array( xss ) 
  Ys = np.array( yss )
  autoencoder.fit(Xs, Ys)

import pickle
def make_dataset():
  
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )

  xss = []
  yss = []
  with open("dataset/corpus.distinct.txt", "r") as f:
    for fi, line in enumerate(f):
      print("now iter ", fi)
      if fi > 1000: 
        break
      line = line.strip()
      head, tail = line.split("___SP___")

      xs = [ [0.]*128 for _ in range(50) ]
      for i, c in enumerate(head): 
        xs[i][c_i[c]] = 1.
      print(np.array( list(reversed(xs)) ).shape)
      xss.append( np.array( list(reversed(xs)) ) )
      
      ys = [ [0.]*128 for _ in range(50) ]
      for i, c in enumerate(tail): 
        ys[i][c_i[c]] = 1.
      yss.append( np.array( ys ) )

  
  Xs = np.array( xss )
  Ys = np.array( yss )
  print(Xs.shape)
  autoencoder.fit( Xs, Ys )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--md' in sys.argv:
    make_dataset()
