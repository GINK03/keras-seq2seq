import bs4
import glob
import os
import sys
import concurrent.futures
import pickle
import MeCab

""" ノクターンノベルズのhtmlから会話コーパスを作成する """
"""
「」が連続すること
片方が、50字以下の会話であること
カタカナのみ
"""
def getKana(chasen):
  es = chasen.split("\n")
  es.pop()
  es.pop()
  return " ".join( map(lambda x:x.split("\t")[1], es) )

def _gen_corpus(arr):
  i, total, name = arr
  sens = []
  if i%100 == 0:
    print("now iter %d / %d"%(i, total) )
  #if i > 200 : 
  #  break
  html = open(name).read()
  soup = bs4.BeautifulSoup(html, "html5lib")
  for t in soup.find_all("div", {"class": "novel_view"}):
    for c in str(t).split("<br/>"):
      sens.append( c.strip() )
  return sens

def gen_corpus():
  files = glob.glob("../nocturne/*")
  total = len(files)
  args  = [ (i, total, name) for i, name in enumerate(glob.glob("../nocturne/*")) ]

  sens  = []
  with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for _sens in executor.map(_gen_corpus, args[:20000]):
      sens += _sens



  m = MeCab.Tagger("-Ochasen")
  with open("corpus.txt", "w") as f:
    for i in range(len(sens) - 2):
      if len(sens[i]) == 0 or len(sens[i+1]) == 0:
        continue
      if sens[i][-1] ==  "」" and \
          sens[i+1][0] == "「":
        head = getKana( m.parse( sens[i] ) )
        tail = getKana( m.parse( sens[i+1] ) )

        if len(head) <= 50 and len(tail) <= 50:
          j = "___SP___".join([head, tail])
          f.write(j + "\n")
          print(j)



""" 128位以下は削る """

def distinct():
  c_f = {}
  for c in open("corpus.txt", "r").read() :
    if c_f.get(c) is None:
      c_f[c] = 0
    c_f[c] += 1
  
  for e, (c, f) in enumerate(sorted( c_f.items(), key=lambda x:x[1]*-1)):
    print(e, c, f)
   
  c_i = {c:e for e, (c,f) in  enumerate(sorted( c_f.items(), key=lambda x:x[1]*-1)[:128]) }

  print(c_i)
  
  open("c_i.pkl", "wb").write( pickle.dumps( c_i ) )
  with open("corpus.txt", "r") as r, open("corpus.distinct.txt", "w") as w:
    for ln in r:
      ln = ln.strip()
      
      distict = "".join( filter(lambda x: x in c_i, list(ln) ) )

      w.write( distict + "\n" ) 

if __name__ == '__main__':
  if '--gen_corpus' in sys.argv:
    gen_corpus()

  if '--distinct' in sys.argv:
    distinct()
