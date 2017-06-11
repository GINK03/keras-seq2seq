from keras.layers import Input, Dense, GRU, LSTM, RepeatVector
from keras.models import Model
from keras.callbacks import LambdaCallback 
import numpy as np
import random
import sys
import pickle
import glob
import copy
timesteps   = 50
inputs  = Input(shape=(timesteps, 128))
#x       = LSTM(512, return_sequences=True)(inputs)
encoded = LSTM(512)(inputs)
encoder = Model(inputs, encoded)

x = RepeatVector(timesteps)(encoded)
x = LSTM(512, return_sequences=True)(x)
x = LSTM(512, return_sequences=True)(x)
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

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  xss = []
  yss = []
  with open("dataset/corpus.distinct.txt", "r") as f:
    for fi, line in enumerate(f):
      print("now iter ", fi)
      if fi >= 3000: 
        break
      line = line.strip()
      head, tail = line.split("___SP___")

      xs = [ [0.]*128 for _ in range(50) ]
      for i, c in enumerate(head): 
        xs[i][c_i[c]] = 1.
      ... #print(np.array( list(reversed(xs)) ).shape)
      xss.append( np.array( list(reversed(xs)) ) )
      
      ys = [ [0.]*128 for _ in range(50) ]
      for i, c in enumerate(tail): 
        ys[i][c_i[c]] = 1.
      yss.append( np.array( ys ) )
  Xs = np.array( xss )
  Ys = np.array( yss )
  print(Xs.shape)
  if '--resume' in sys.argv:
    model = sorted( glob.glob("models/*.h5") ).pop(0)
    print("loaded model is ", model)
    autoencoder.load_weights(model)
  for i in range(10):
    
    print_callback = LambdaCallback(on_epoch_end=callbacks)
    batch_size = random.randint( 4, 64 )
    autoencoder.fit( Xs, Ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    autoencoder.save("models/%9f_%09d.h5"%(buff['loss'], i))
    print("saved ..")
    print("logs...", buff )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()
