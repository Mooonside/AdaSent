"""
    An implementation of
        Self-Adaptive Hierarchical Sentence Model
    Handed By
        Yifeng Chen
"""
import tensorflow as tf
import numpy as np
import os
from time import gmtime, strftime, time
from CCF_Reader import CCF_Reader
from Weibo_Reader import Weibo_Reader
import re


class AdaSent:
  def __init__(self, name='AdaSent'):
    self.name = name

    # Model Path Config
    self.path_para = './' + self.name + '_Para'
    if not os.path.exists(self.path_para):
      os.makedirs(self.path_para)

    self.path_log = './' + self.name + '_Log'
    if not os.path.exists(self.path_log):
      os.makedirs(self.path_log)

    # TF Session Config
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    self.sess = tf.Session(config=config)
    # self.sess = tf.Session()
    # Log File Config
    self.flog = open(self.name + '_log.txt', 'a')
    self.flog.write("--- LOG INFO : %s ---\n" % strftime("%Y-%m-%d %H:%M:%S", gmtime()))

  def set_reader(self, reader):
    self.reader = reader

  def logger(self, mesg):
    self.flog.write("{}".format(mesg))
    print(mesg)

  def apply_vars(self, N, B, W, S):
    weight_init = tf.truncated_normal_initializer(stddev=1e-3)

    wemb = tf.get_variable(
      name="WordEmbd",
      shape=[B, W],
      trainable=True,
      initializer=tf.constant_initializer(self.reader.EMB),
      dtype=tf.float32
    )

    semb = tf.get_variable(
      name="SentEmbd",
      shape=[W, S],
      trainable=True,
      initializer=weight_init,
      dtype=tf.float32
    )

    comb_left = tf.get_variable(
      name="CombLeft",
      shape=[S, S],
      trainable=True,
      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
      dtype=tf.float32
    )

    comb_right = tf.get_variable(
      name="CombRight",
      shape=[S, S],
      trainable=True,
      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
      dtype=tf.float32
    )

    comb_bias = tf.get_variable(
      name="CombBias",
      shape=[S],
      trainable=True,
      initializer=tf.zeros_initializer,
      dtype=tf.float32
    )

    gate_left = tf.get_variable(
      name="GateLeft",
      shape=[S, 3],
      trainable=True,
      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
      dtype=tf.float32
    )

    gate_right = tf.get_variable(
      name="GateRight",
      shape=[S, 3],
      trainable=True,
      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
      dtype=tf.float32
    )

    gate_bias = tf.get_variable(
      name="GateBias",
      shape=[3],
      trainable=True,
      initializer=tf.zeros_initializer,
      dtype=tf.float32
    )

    classifier_p = tf.get_variable(
      name="ClsfPara",
      shape=[S, 2],
      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
      dtype=tf.float32
    )

    voter_p = tf.get_variable(
      shape=[S, 1],
      name="VotePara",
      trainable=True,
      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
      dtype=tf.float32
    )

    sents = tf.placeholder(
      dtype=tf.int32,
      shape=[None, N],
      name="Sentences"
    )

    labels = tf.placeholder(
      dtype=tf.int32,
      shape=[None],
      name="Labels"
    )

    self.global_step = tf.get_variable(
      name='global_step',
      initializer=tf.zeros_initializer,
      shape=[],
      dtype=tf.int64,
      trainable=False
    )

    def frobenius_norm(tensor):
      return tf.trace(tf.transpose(tensor) * tensor)

    self.norm_loss = frobenius_norm(comb_left) + frobenius_norm(comb_right)

    self.var_dict = {
      "WordEmbd": wemb,
      "SentEmbd": semb,
      "CombLeft": comb_left,
      "CombRight": comb_right,
      "CombBias": comb_bias,
      "GateLeft": gate_left,
      "GateRight": gate_right,
      "GateBias": gate_bias,
      "ClsfPara": classifier_p,
      "VotePara": voter_p,
      "Sentences": sents,
      "Labels": labels,
      "global_step": self.global_step
    }

  def build(self, N, B, W, S):
    """
            build the model

            N : sentence length
            B : word bank num
            W : word embedding dimension 
            S : sentence embedding dimension 
        """
    self.debug_dict = {}
    self.global_step = self.var_dict['global_step']
    sents = self.var_dict['Sentences']
    labels = self.var_dict['Labels']
    # <editor-fold desc="Apply For Parameters">
    wemb = self.var_dict['WordEmbd']
    semb = self.var_dict['SentEmbd']
    comb_left = self.var_dict['CombLeft']
    comb_right = self.var_dict['CombRight']
    comb_bias = self.var_dict['CombBias']
    gate_left = self.var_dict['GateLeft']
    gate_right = self.var_dict['GateRight']
    gate_bias = self.var_dict['GateBias']
    classifier_p = self.var_dict['ClsfPara']
    voter_p = self.var_dict['VotePara']
    # </editor-fold>

    # Extract Word Embedding, Shape : [None, N, W]
    sents_wemb = tf.nn.embedding_lookup(wemb, sents, name="SentWemb")
    # sents_wemb = tf.tensordot(sents, wemb, axes=[[1], [0]], name="SentWemb")

    # Extract Sentence Embedding,Shape : [None, N, S]
    sents_semb = tf.tensordot(sents_wemb, semb, axes=[[2], [0]], name="SentSemb")

    # <editor-fold desc="Generate Pyramid">
    i_final = sents_semb
    pyrmaid = []

    # Shape : [None, S]
    i_pool = tf.reduce_max(
      input_tensor=i_final,
      axis=1,
      keep_dims=False,
      name=('APool_%d' % 0)
    )
    pyrmaid.append(i_pool)

    for i in range(1, N):
      # Pyramid Arch
      # Shape : [None, N-i, S]
      i_hleft = tf.slice(
        input_=i_final,
        begin=[0, 0, 0],
        size=[-1, N - i, -1]
      )
      # Shape : [None, N-i, S]
      i_hright = tf.slice(
        input_=i_final,
        begin=[0, 1, 0],
        size=[-1, N - i, -1]
      )
      # Shape : [None, N-i, S] . [S, S] => [None, N-i, S]
      i_sleft = tf.tensordot(i_hleft, comb_left, axes=[[2], [0]], name=("LComb_%d" % i))
      i_sright = tf.tensordot(i_hright, comb_right, axes=[[2], [0]], name=("RComb_%d" % i))
      # print(i_sleft,i_sright)
      # Shape : [None, N-i, S]
      i_scomb = tf.nn.tanh(
        i_sleft + i_sright + comb_bias,
        name=("AComb_%d" % i)
      )

      # Shape : [None, N-i, S] . [S, 3] => [None, N-i, 3]
      i_gleft = tf.tensordot(i_hleft, gate_left, axes=[[2], [0]], name=("LGate_%d" % i))
      i_gright = tf.tensordot(i_hright, gate_right, axes=[[2], [0]], name=("RGate_%d" % i))
      # print(i_gleft, i_gright)

      # Shape [None, N-i, 3, 1]
      i_gcomb = tf.expand_dims(
        input=tf.nn.softmax(
          logits=i_gleft + i_gright + gate_bias,
          dim=2
        ),
        axis=-1,
        name=("AGate_%d" % i)
      )
      self.debug_dict["AGate_%d" % i] = i_gcomb
      # print(i_gcomb)

      # Shape : [None, S]
      i_pool = tf.reduce_max(
        input_tensor=tf.squeeze(
          input=tf.matmul(
            # [None, N-i, S, 3]
            tf.stack(
              values=[i_sleft, i_sright, i_scomb],
              axis=-1),
            # [None, N-i, 3, 1]
            i_gcomb
          ),
          axis=3,
        ),
        axis=1,
        keep_dims=False,
        name='APool_%d' % i
      )
      pyrmaid.append(i_pool)
    # </editor-fold>

    # pyramid has shape : [None ,S] * N, reshape to [None, N, S]
    pyrmaid = tf.stack(pyrmaid, axis=1)
    self.debug_dict['pyramid'] = pyrmaid
    # Gating Network
    # shape : [None, N, 1]

    # [None, N, S] . [S, 1] => [None, N, 1]
    level_attention = tf.tensordot(
      pyrmaid,
      voter_p,
      axes=[[2], [0]]
    )
    self.debug_dict['LevelAttention'] = level_attention

    voter_s = tf.nn.softmax(
      # Shape : [None, N, 1]
      logits=level_attention,
      dim=1,
      name="VoterS"
    )
    self.debug_dict['GateingNet'] = voter_s
    self.debug_dict['VoterP'] = voter_p

    level_clsf = tf.tensordot(pyrmaid, classifier_p, axes=[[2], [0]])
    level_clsf = tf.reshape(level_clsf, [-1, 2])
    voter_s = tf.reshape(voter_s, [-1, 1])

    score = tf.reshape(level_clsf * voter_s, [-1, N, 2])
    self.debug_dict['LevelScores'] = score

    # [None, 2]
    score = tf.reduce_sum(
      score,
      axis=1,
      keep_dims=False
      # ,name="Score"
    )
    #
    # Use Structured Sentence as Input
    score = tf.reduce_mean(
      score,
      axis=0,
      keep_dims=True,
      name="Scores"
    )
    self.debug_dict['FinalScore'] = score

    # [None]
    self.prediction = tf.cast(tf.argmax(input=score, axis=1), tf.int32, name="Prediction")
    # [None, 2]
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      labels=onehot_labels, logits=score))
    self.loss += 1e-3 * self.norm_loss
    self.score = score
    return score

  def valid(self, MAX_SENT_LEN, BATCH_SIZE):
    """
        valid the model
    """
    VALID_NUM_PER_EPOCH = len(self.reader.valid_idx)
    # valid_debug_file = open("valid_debug_file.txt", 'w')

    cnt = 0
    tloss = 0

    tic = time()
    for sent_id in range(0, VALID_NUM_PER_EPOCH, BATCH_SIZE):
      # sentences, label = self.reader.extract_one_sentence(sent_id, MAX_SENT_LEN, 'VALID')
      sentences, label, raw = self.reader.extract_one_sentence(sent_id, MAX_SENT_LEN, 'VALID')

      feed_dict = {
        'Sentences:0': sentences,
        'Labels:0': label
      }

      loss, pred = self.sess.run([self.loss, self.prediction], feed_dict)

      tloss += np.sum(loss)
      cnt += np.sum(label == pred)

    toc = time()
    avg_loss = tloss / VALID_NUM_PER_EPOCH
    avg_acc = cnt / VALID_NUM_PER_EPOCH

    self.logger("CROSS VALIDATION: loss{:.3f}, accuracy {:.3f} in {} s"
                .format(avg_loss, avg_acc, toc - tic))
    self.save_model(avg_acc)

  def save_model(self, val_acc, path=None):
    """
        save the model to path
    """
    if path is None:
      path = self.path_para
    if val_acc >= self.best_performance():
      pass
    else:
      return

    path = self.saver.save(
      self.sess,
      path + "/VA-{:.4f}.mds".format(val_acc),
      global_step=self.global_step
    )
    self.logger("Save models to {}.\n".format(path))
    self.logger("---End of LOGGING--- %s\n" % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # self.flog.close()
    pass

  def best_performance(self, path=None):
    ckpt = tf.train.get_checkpoint_state(self.path_para)
    if ckpt is not None:
      path = ckpt.model_checkpoint_path
      va = re.search('0.[0-9]*', path)
      return float(va.group(0))
    else:
      return 0

  def load_model(self, path=None):
    """
            load the model from path
    """
    self.saver = tf.train.Saver(max_to_keep=3)
    if path is None:
      path = self.path_para
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt is not None:
      self.logger('Loading from {}.\n'.format(ckpt.model_checkpoint_path))
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
      self.logger('No Previous Models Found In {}!\n'.format(path))
      self.logger('Using Xavier Initialization\n')
      self.sess.run(tf.global_variables_initializer())

  @staticmethod
  def info():
    print('About The Model:')
    print('- Python     : 3.5.1')
    print('- Tensorflow : 1.4.0')

  def train(self, epoch_num=1, BATCH_SIZE=64, MAX_SENT_LEN=40, SET_DIM=500, CV_INT=256, CV_K=10, test_reader=None):
    if self.reader is None:
      print("No Data Reader is Allocated!\n")
      return

    EMD_DIM = self.reader.EMBEM_DIM
    WORD_NUM = len(self.reader.WBANK.keys())

    self.reader.k_fold(K=CV_K)
    TRAIN_NUM_PER_EPOCH = len(self.reader.train_idx)

    self.apply_vars(MAX_SENT_LEN, WORD_NUM, EMD_DIM, SET_DIM)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    score = self.build(MAX_SENT_LEN, WORD_NUM, EMD_DIM, SET_DIM)
    grad_vars = optimizer.compute_gradients(self.loss)
    self.grad_vars = [
      (tf.clip_by_norm(grad, clip_norm=10), var)
      for grad, var in grad_vars if grad is not None
    ]
    self.grad_dict = dict([
      [var.name, grad]
      for grad, var in self.grad_vars
    ])

    self.train_op = optimizer.apply_gradients(self.grad_vars, self.global_step)
    self.load_model()

    for epoch_id in range(epoch_num):
      for sent_id in range(0, TRAIN_NUM_PER_EPOCH, BATCH_SIZE):
        tic = time()
        # should be change
        # sentences, label = self.reader.extract_one_sentence(sent_id, MAX_SENT_LEN, 'Train')
        sentences, label, _ = self.reader.extract_one_sentence(sent_id, MAX_SENT_LEN, 'Train')
        feed_dict = {
          'Sentences:0': sentences,
          'Labels:0': label
        }

        loss, pred, _ = self.sess.run([self.loss, self.prediction, self.train_op], feed_dict)
        toc = time()
        acc = np.sum(pred == label) / BATCH_SIZE

        self.logger("Epoch {} Step {} Loss {:.3f} Acc: {:.3f}"
                    .format(epoch_id, sent_id, np.mean(loss), acc))

        if sent_id % CV_INT == 0 or epoch_id == TRAIN_NUM_PER_EPOCH - 1:
          # do cross validation
          def debug_log(mdict, feed_dict, outname):
            debug_grads = [key for key in mdict.keys()]
            debug_items = [mdict[key] for key in debug_grads]
            debug_v = self.sess.run(debug_items, feed_dict)
            debug_dict = dict([[debug_grads[idx], debug_v[idx]] for idx in range(len(debug_v))])
            debug_file = open(outname, 'a')
            debug_file.write('=' * 8 + str(sent_id) + '=' * 8 + '\n')
            for i in debug_dict.keys():
              debug_file.write(i + '\n')
              debug_file.write('-' * 8)
              try:
                item = np.asarray(debug_dict[i])
                debug_file.write("mean:{} , var{}, max{} \n".format(np.mean(item), np.var(item), np.max(item)))
              except:
                item = np.asarray(debug_dict[i][0])
                debug_file.write("mean:{} , var{} max {} \n".format(np.mean(item), np.var(item), np.max(item)))
            debug_file.close()

          debug_log(self.grad_dict, feed_dict, 'grad_log.txt')
          debug_log(self.debug_dict, feed_dict, 'runtime_log.txt')

          self.logger("VALIDATION at Epoch {} Step {}".format(epoch_id, sent_id))
          if test_reader is None:
            self.valid(MAX_SENT_LEN, BATCH_SIZE)
          else:
            self.test(test_reader, MAX_SENT_LEN, SET_DIM)

  def test(self, test_reader, MAX_SENT_LEN, SET_DIM, IS_BUILD=True, outname="test_out.txt"):
    test_reader.k_fold(K=0)
    TEST_SENT_NUM = len(test_reader.idx_sentences)
    ts, tp, tl = 0, 0 ,0

    if not IS_BUILD:
      EMD_DIM = self.reader.EMBEM_DIM
      WORD_NUM = len(self.reader.WBANK.keys())
      self.apply_vars(MAX_SENT_LEN, WORD_NUM, EMD_DIM, SET_DIM)
      score = self.build(MAX_SENT_LEN, WORD_NUM, EMD_DIM, SET_DIM)
      self.sess.run(tf.global_variables_initializer())
      self.load_model()

    test_debug_file = open('test_debug_file.txt','w')
    output_file = open(outname,'w')
    output_file.write("<weibos>\n")
    for sid in range(0, TEST_SENT_NUM):
      sent, label , raw = test_reader.extract_one_sentence(idx=sid, slen=MAX_SENT_LEN)
      feed_dict = {
        'Sentences:0': sent,
      }
      #[?, 2], [?, 1], []
      p = self.sess.run(self.prediction, feed_dict)
      if int(p) == 1:
        output_file.write("\t<weibo id=\"%d\" polarity=\"%d\">%s</weibo>\n"%
                        (sid, 1,test_reader.raw_docs[sid]))
      else:
        output_file.write("\t<weibo id=\"%d\" polarity=\"%d\">%s</weibo>\n"%
                        (sid, -1,test_reader.raw_docs[sid]))
      if label != p:
        test_debug_file.write("Label : {} Predict : {}\n".format(label, p))
        for sent in raw:
          test_debug_file.write(sent)
        test_debug_file.write('\n')
      else:
        tp += 1

    tp /= TEST_SENT_NUM
    tl /= TEST_SENT_NUM
    print("Test Loss {:.3f} Test ACC {:.3f}".format(tl, tp))
    output_file.write("</weibos>")


if __name__ == "__main__":
  model = AdaSent()
  # first to English?
  train_reader = CCF_Reader()
  train_reader.read_doc('./dataset/en_sample_data/sample.negative.txt')
  train_reader.read_doc('./dataset/en_sample_data/sample.positive.txt')

  # CCF_Reader = CCF_Reader('cn_format.txt',None)
  # CCF_Reader.read_doc('cn_review_reserved_div_sentence.txt')
  train_reader.tokenize()
  train_reader.load_emb(['glove.6B.100d.txt'])
  print(train_reader.EMB.shape)
  model.set_reader(train_reader)

  # test_reader = train_reader
  test_reader = Weibo_Reader(mode='EN')
  test_reader.WBANK = train_reader.WBANK

  test_reader.read_doc('dataset/sample_en.xml')
  test_reader.tokenize(mode='test')
  model.test(test_reader, MAX_SENT_LEN=20, SET_DIM=500, IS_BUILD=False, outname='task2output.xml')
  # test_reader.WBANK = train_reader.WBANK
  # test_reader.read_doc('./dataset/test.en.txt')
  # test_reader.tokenize(mode='test')
  # use whole training data!
  # model.train(epoch_num=2, BATCH_SIZE=1, MAX_SENT_LEN=20,CV_K=10,test_reader=None)
  # model.train(epoch_num=2, BATCH_SIZE=1, MAX_SENT_LEN=20,CV_K=0,test_reader=test_reader)
  # model.test(test_reader, MAX_SENT_LEN=20, SET_DIM=500, IS_BUILD=False)