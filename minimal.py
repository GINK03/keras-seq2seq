from keras.layers     import Input, Dense, GRU, LSTM, RepeatVector
from keras.models     import Model
from keras.callbacks  import LambdaCallback 
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re
timesteps   = 50
inputs  = Input(shape=(timesteps, 128))
encoded = LSTM(512)(inputs)
encoder = Model(inputs, encoded)

x = RepeatVector(timesteps)(encoded)
x = Bi(LSTM(512, return_sequences=True))(x)
#x = LSTM(512, return_sequences=True)(x)
decoded = TD(Dense(128, activation='softmax'))(x)

autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer=Adam(), loss='categorical_crossentropy')

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
      if fi >= 100000: 
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

    """ 確実に更新するため、古いデータは消す """
    #os.system("rm models/*")
  for i in range(1000):
    
    print_callback = LambdaCallback(on_epoch_end=callbacks)
    batch_size = random.randint( 32, 64 )
    random_optim = random.choice( [Adam(), SGD(), RMSprop()] )
    #print( vars( autoencoder.optimizer.lr  ) )
    print( random_optim )
    autoencoder.optimizer = random_optim
    #sys.exit()
    autoencoder.fit( Xs, Ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    autoencoder.save("models/%9f_%09d.h5"%(buff['loss'], i))
    print("saved ..")
    print("logs...", buff )

def predict():
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c = { i:c for c, i in c_i.items() }
  xss = []
  heads = []
  with open("dataset/corpus.distinct.txt", "r") as f:
    for fi, line in enumerate(f):
      print("now iter ", fi)
      if fi >= 1000: 
        break
      line = line.strip()
      head, tail = line.split("___SP___")
      heads.append( head ) 
      xs = [ [0.]*128 for _ in range(50) ]
      for i, c in enumerate(head): 
        xs[i][c_i[c]] = 1.
      xss.append( np.array( list(reversed(xs)) ) )
    
  Xs = np.array( xss[:128] )
  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  autoencoder.load_weights(model)

  Ys = autoencoder.predict( Xs ).tolist()
  for head, y in zip(heads, Ys):
    terms = []
    for v in y:
      term = max( [(s, i_c[i]) for i,s in enumerate(v)] , key=lambda x:x[0])[1]
      terms.append( term )
    tail = re.sub(r"」.*?$", "」", "".join( terms ) )
    print( head, "___SP___", tail )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
