import re
import numpy as np
from Weibo_Reader import Weibo_Reader

from Thu_Split.USEthulac import thu_split

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
NUMBER = "0123456789"


def transform(word):
  trans = []
  if word.endswith('es'):
    trans.append(word[:-2])
  if word.endswith('s'):
    trans.append(word[:-1])
  if word.endswith('ing'):
    trans.append(word[:-3])
  if word.endswith('ied'):
    trans.append(word[:-3] + 'y')
  if word.endswith('ed'):
    trans.append(word[:-2])
    trans.append(word[:-1])
  if word.endswith('ly'):
    trans.append(word[:-2])
  return trans


def clean_wd(word):
  if word in ['\'s', '\'ll', 'n\'t', '\'re', '\'em', '\'ve', '\'d', '\'v', '\'m']:
    return word
  return word.strip(' |-+_=@#$%^&*\\?<>{}[]()`~\'\".')


def clean_sents(sents):
  sents = sents.replace('。', '.')
  sents = sents.replace('，', ',')
  sents = sents.replace('：', ';')
  sents = sents.replace('！', '!')
  sents = sents.replace('？', '?')

  sents = re.sub('(mrs|mr|ms|ps|p\.s|dr)\.', ' ', sents)
  sents = re.sub('\.\.+', ' ELLIPSE ', sents)

  sents = re.sub('[^ ]*\.net', ' ', sents)
  sents = re.sub('[^ ]*\.com', ' ', sents)
  sents = re.sub('[^ ]*\.org', ' ', sents)

  sents = sents.replace('\n', '')
  sents = sents.replace('\t', '')
  sents = sents.replace('. ', ' . ')

  sents = re.sub('[ ]*,[ ]*', ' , ', sents)
  sents = re.sub('[ ]*&+[ ]*', ' & ', sents)
  sents = re.sub('[ ]*\$+[ ]*', ' $ ', sents)
  sents = re.sub('[ ]*\\+[ ]*', ' or ', sents)
  sents = re.sub('[ ]*/+[ ]*', ' or ', sents)
  sents = re.sub('[ ]*%+[ ]*', ' % ', sents)
  sents = re.sub('[ ]*_+[ ]*', ' _ ', sents)
  sents = re.sub('[ ]*-+[ ]*', ' - ', sents)
  sents = re.sub('[ ]*\(+[ ]*', ' ( ', sents)
  sents = re.sub('[ ]*\)+[ ]*', ' ) ', sents)
  sents = re.sub('[ ]*\[+[ ]*', ' [ ', sents)
  sents = re.sub('[ ]*\]+[ ]*', ' ] ', sents)
  sents = re.sub('[ ]*\{+[ ]*', ' { ', sents)
  sents = re.sub('[ ]*\}+[ ]*', ' } ', sents)
  sents = re.sub('[ ]*!+[ ]*', ' ! ', sents)
  sents = re.sub('[ ]*\?+[ ]* ', ' ? ', sents)
  sents = re.sub('[ ]*:+[ ]*', ' : ', sents)
  sents = re.sub('[ ]*;+[ ]*', ' ; ', sents)
  sents = re.sub('[ ]*~+[ ]*', ' ~ ', sents)
  sents = re.sub('[ ]*`+[ ]*', ' ` ', sents)
  sents = re.sub('\'s', ' \'s ', sents)
  sents = re.sub('\"s', ' \"s ', sents)
  sents = re.sub('n\'t', ' n\'t ', sents)
  sents = re.sub('\'ll', ' \'ll ', sents)
  sents = re.sub('\'re', ' \'re ', sents)
  sents = re.sub('\'d', ' \'d ', sents)
  sents = re.sub('\'ve', ' \'ve ', sents)
  sents = re.sub('\'v', ' \'v ', sents)
  sents = re.sub('\'m', ' \'m ', sents)
  sents = re.sub(' ELLIPSE ', ' ... ', sents)

  sents = sents.replace('=', '')
  sents = sents.replace('+', '')
  return sents


def div_sentence(sentlist):
  for punc in [' . ', ' ? ', ' ! ', ' ; ', ' ~ ', ' ... ']:
    divisions = []
    for sub in sentlist:
      splits = sub.split(punc)
      for idx, item in enumerate(splits):
        item = item.strip(" ")
        if item != "":
          if idx == len(splits) - 1:
            divisions.append(item)
          else:
            divisions.append(item + punc)
    sentlist = divisions
  sentlist = [i for i in sentlist if i not in ['.', '?', '!', ';', '~', '...']]
  # remove stuff likes 1. xxxx
  sentlist = [i for i in sentlist if re.sub("[0-9]+[ ]*\.[ ]*", "", i) != ""]
  return sentlist


class CCF_Reader:
  def __init__(self):
    self.WBANK = {'_UNK': 0, "_PAD": 1}
    self.idx_sentences = []
    self.raw_sentences = []
    self.doc_len = []
    self.labels = []
    self.raw_docs = []

  def load_emb(self, emb_paths, mode="MINIMAL"):
    """
    Load Pre-trained Word Embeddings, add words into WordBanks
    :param emb_path: 
    :return:None 
    """
    GLOVE_EMB = []
    WMAP = {}
    cnt = 0

    for emb_path in emb_paths:
      S_EMB = []
      emb_file = open(emb_path)
      # line = emb_file.readline().rstrip('\n')
      # WORD_NUM, EMBEM_DIM = [int(i) for i in line.split(' ')]
      for line in emb_file.readlines():
        splits = line.rstrip('\n').split(' ')
        word = splits[0]
        if not word in WMAP.keys():
          WMAP[word] = cnt

        emb = [float(i) for i in splits[1:]]
        S_EMB.append(emb)
        cnt += 1
      S_EMB = np.asarray(S_EMB)
      mean, var = np.mean(S_EMB), np.var(S_EMB)
      print('{} GLOVE_EMB: mean {:.3f} var:{:.3f}'.format(emb_path, mean, var))
      GLOVE_EMB.append(S_EMB)
      emb_file.close()

    GLOVE_EMB = np.concatenate(GLOVE_EMB, axis=0)

    self.WBANK_NUM = len(self.WBANK.keys())
    self.EMBEM_DIM = GLOVE_EMB.shape[1]
    self.EMB = np.random.randn(self.WBANK_NUM, self.EMBEM_DIM) * np.sqrt(var) + mean

    f = open('glove_miss.txt', 'w')
    hits = 0
    for i in self.WBANK.keys():
      find = False
      if i in WMAP.keys():
        self.EMB[self.WBANK[i], :] = GLOVE_EMB[WMAP[i], :]
        hits += 1
        continue

      trans = transform(i)
      for t in trans:
        if t in WMAP.keys():
          self.EMB[self.WBANK[i], :] = GLOVE_EMB[WMAP[t], :]
          hits += 1
          find = True
          break

      if not find:
        f.write(i + '\n')
        # not found here
    f.close()
    print('EMB: {} * {}-d with {:.2f}% hits'
        .format(self.WBANK_NUM, self.EMBEM_DIM, hits * 100 / self.WBANK_NUM))

  def add_word(self, word):
    """
    Add a new word to word bank
    :param word: 
    :return: None
    """
    if word not in self.WBANK.keys():
      self.WBANK[word] = len(self.WBANK.keys())
    return self.WBANK[word]

  def read_doc(self, doc_path, sid=0):
    """
    Turn words in doc into token_ids, meanwhile add untrained words into word bank.
    :param  doc_path: 
    :return: None
    """
    doc_file = open(doc_path)
    raw_sentences = []
    # debug = open('debug.txt', 'a')
    while True:
      # read when met with < Review
      while True:
        line = doc_file.readline()
        if line == "":
          break
        if re.search('<review id=\"[0-9]+\">',line) is not None:
          break
      if line != "":
        id = int(re.search('[+-]?\d+', line).group(0))
        # if id != sid:
        #   print(id)
        sid += 1
      else:
        break
      # print(id.group(0))
      sents = ""
      while True:
        line = doc_file.readline().strip('\n')
        if line == "</review>":
          break
        if len(line) > 0 and (line[-1] in ALPHABET or line[-1] in NUMBER):
          line += '.'
        # read in several \n and add on
        line += " "
        sents += line
      sents = sents.lower()
      sents = clean_sents(sents)
      self.raw_docs.append(sents)

      # divide into sub sentences
      splits = div_sentence([sents])
      if len(splits) == 0:
        splits = [sents]
      raw_sentences.append(splits)
    doc_file.close()

    self.raw_sentences += raw_sentences

    SENTENCE_NUM = len(raw_sentences)
    print("Loaded {} Sentences".format(SENTENCE_NUM))

    if doc_path.__contains__('pos'):
      self.labels = np.concatenate([self.labels, np.ones(SENTENCE_NUM)])
    else:
      self.labels = np.concatenate([self.labels, np.zeros(SENTENCE_NUM)])

  def tokenize(self, mode='train'):
    if mode == 'train':
      self.idx_sentences = []
      for sent in self.raw_sentences:
        sents = []
        for subsent in sent:
          subs = []
          for word in subsent.split(' '):
            word = clean_wd(word)
            subs.append(self.add_word(word))
          sents.append(subs)
        self.idx_sentences.append(sents)
    else:
      self.idx_sentences = []
      misses = {}
      for sent in self.raw_sentences:
        sents = []
        for subsent in sent:
          subs = []
          for word in subsent.split(' '):
            word = clean_wd(word)
            if word in self.WBANK.keys():
              subs.append(self.add_word(word))
            else:
              misses[word] = 1
              subs.append(self.WBANK['_UNK'])
          sents.append(subs)
        self.idx_sentences.append(sents)
      print('{:.3f}% Unknown'.format(len(misses)*100 / len(self.WBANK)))

    sentence_stats = [len(j) for i in self.idx_sentences for j in i]
    print("Max Len : {}, Min Len : {}, Avg Len : {}".format(
      np.max(sentence_stats), np.min(sentence_stats), np.mean(sentence_stats)
    ))


  def flat_structure(self):
    sp = []
    for idx, label in enumerate(self.labels):
      sp.append([label for _ in range(len(self.idx_sentences[idx]))])
    self.labels = [j for i in sp for j in i]
    self.idx_sentences = [j for i in self.idx_sentences for j in i]
    self.raw_sentences = [j for i in self.raw_sentences for j in i]

  def save_to_txt(self, fname='WordBank.txt'):
    out = open(fname, 'w')
    for i in self.WBANK.keys():
      out.write(i + '\n')
    out.close()

  def pad_cut(self, sent, bound):
    init_len = len(sent)
    for j in range(init_len, bound):
      sent.append(self.WBANK['_PAD'])
    sent = sent[:bound]
    return sent

  def k_fold(self, K=10):
    # should keep the distributions
    # first read in 5000 positive then 5000 negative
    SENTENCE_NUM = len(self.raw_sentences)
    # all for valid
    if K == 0:
      self.valid_idx = []
      self.train_idx = np.arange(SENTENCE_NUM)
      return

    division = SENTENCE_NUM // K

    ids = np.arange(SENTENCE_NUM)
    ids = np.random.permutation(ids)

    self.valid_idx = ids[:division]
    self.train_idx = ids[division:]

  def extract_n_sentence(self, idx, n, slen=20, mode='Train'):
    if mode == 'Train':
      end = min(idx + n, len(self.train_idx))
      train_idx = self.train_idx[idx:end]
      sent, raw, lbs = [], [], []
      for idx in train_idx:
        sent.append(self.idx_sentences[idx])
        raw.append(self.raw_sentences[idx])
        lbs.append(self.labels[idx])
    else:
      end = min(idx + n, len(self.valid_idx))
      valid_idx = self.valid_idx[idx:end]
      sent, raw, lbs = [], [], []
      for idx in valid_idx:
        sent.append(self.idx_sentences[idx])
        raw.append(self.raw_sentences[idx])
        lbs.append(self.labels[idx])
    for i in range(len(sent)):
      sent[i] = self.pad_cut(sent[i], slen)
    return sent, lbs, raw

  def extract_one_sentence(self, idx, slen=20, mode='Train'):
    """
    :param idx: idx used sequentially for training
    :param mode: 
    :return: 
    """
    if mode == 'Train':
      sent = self.idx_sentences[self.train_idx[idx]]
      raw = self.raw_sentences[self.train_idx[idx]]
      lbs = int(self.labels[self.train_idx[idx]])
    else:
      sent = self.idx_sentences[self.valid_idx[idx]]
      raw = self.raw_sentences[self.valid_idx[idx]]
      lbs = int(self.labels[self.valid_idx[idx]])
    for i in range(len(sent)):
      sent[i] = self.pad_cut(sent[i], slen)
    return sent, [lbs], raw

  def ratio(self, slen):
    lens = np.asarray([len(j) for i in self.idx_sentences for j in i])
    return np.mean(lens > slen)




if __name__ == "__main__":
  train_reader = CCF_Reader()
  # train_reader.read_doc('./dataset/en_sample_data/sample.positive.txt',sid=0)
  # train_reader.read_doc('./dataset/en_sample_data/sample.negative.txt',sid=5000)
  train_reader.read_doc('dataset/thu_cn/thu_cn_div_neg_sentence.txt')
  train_reader.read_doc('dataset/thu_cn/thu_cn_div_pos_sentence.txt')
  train_reader.tokenize(mode='train')
  # train_reader.load_emb(emb_paths=['thu_cn_wiki_format_400_.txt'])
  train_reader.load_emb(emb_paths=['thu_cn_100d.txt'])
  print(train_reader.EMB.shape)
  # train_reader.load_emb(emb_path='./glove.6B.100d.txt')
  # train_reader.flat_structure()
  train_reader.k_fold(K=10)

  test_reader = Weibo_Reader(mode='CHN')
  # train_reader.read_doc('./dataset/en_sample_data/sample.positive.txt',sid=0)
  # train_reader.read_doc('./dataset/en_sample_data/sample.negative.txt',sid=5000)
  test_reader.read_doc('dataset/task2input.xml')
  test_reader.EMB = train_reader.EMB
  test_reader.WBANK = train_reader.WBANK
  test_reader.tokenize(mode='test')

  # debug = open('ccf_out.txt','w')
  # for i in train_reader.raw_sentences:
  #   for j in i:
  #     debug.write(j+'\n')
  #   debug.write('----------\n')
  # debug.close()

  # sents, lbs, raw = train_reader.extract_one_sentence(idx=0, slen=20)
  # print(sents)
  # print(lbs)
  # print(raw)
  #
  # print("{:.3f}%  Reviews will be Cut!".format(train_reader.ratio(10)*100))

  # test_reader = CCF_Reader()
  # test_reader.WBANK = train_reader.WBANK
  # test_reader.read_doc('./dataset/test.en.txt')
  # test_reader.tokenize(mode='test')
  #
  # # do no folding
  # test_reader.k_fold(0)
  # sents, lbs, raw = test_reader.extract_one_sentence(idx=0, slen=20, mode='train')
  # print(sents)
  # print(lbs)
  # print(raw)