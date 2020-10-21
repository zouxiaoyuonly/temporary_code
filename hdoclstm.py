# coding=utf-8
# ######################################################################################################################
# 说明：文章句子标注任务
# 文件：HierarchicalLstm.py
# 时间：2019.10.17
# 版本：1.0.0
# ######################################################################################################################
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint,Callback
from keras.optimizers import Adam
import os
from tqdm import tqdm
import json
import pickle
import time
# from keras_contrib.layers.crf import CRF
import numpy
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import load_model
from itertools import chain
from matplotlib import pyplot
# from keras_contrib.layers import CRF

class HLSTM():
    def __init__(self,dictpath = './config/vocabdict.pkl',word2vecpath = None):
        self.max_sents = 75      # 句子个数
        self.max_sent_len = 300  # 句子长度
        self.embedding_dim = 300
        self.lstm_dim = 100
        self.batch_size = 8
        self.epochs = 10
        self.dictpath = dictpath
        self.word2vecpath = word2vecpath
        self.modeldir = './model'
        self.labels = ['begin','mid','end','ad']
        self.tag_count = len(self.labels)
        self.vocabs = self.read_dictionary(self.dictpath)
        self.tag2id,self.id2tag = self.get_labels()
        self.tags =  {
            'begin':'文章首部',
            'mid':'文章中间',
            'end':'中间尾部',
            'ad':'广告',
        }

    '''构建模型结构'''
    def createmodel(self):
        vocabs = self.vocabs
        embedding_matrix = self.create_matrix(vocabs,self.word2vecpath)
        embedding_layer = Embedding(len(vocabs) + 1,
                                    self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_sent_len,
                                    trainable=True)

        sentence_input = Input(shape=(self.max_sent_len,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(LSTM(self.lstm_dim))(embedded_sequences)
        sentEncoder = Model(sentence_input, l_lstm)

        review_input = Input(shape=(self.max_sents,self.max_sent_len), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)

        l_lstm_sent = Bidirectional(LSTM(self.lstm_dim, return_sequences=True))(review_encoder)
        output = TimeDistributed(Dense(self.tag_count, activation='softmax'))(l_lstm_sent)

        model = Model(review_input, output)
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
        model.summary()
        return model

    '''构建嵌入层'''
    def create_matrix(self,vocab,word2vecpath):
        wordvec = self.load_wordvec(vocab,word2vecpath)
        if wordvec == None:
            embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab)+1, self.embedding_dim))
            embedding_matrix = np.float32(embedding_matrix)
        else:
            embedding_matrix = np.random.random((len(vocab) + 1, self.embedding_dim))
            for word, i in vocab.items():
                embedding_vector = wordvec.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix

    '''读取字典'''
    def read_dictionary(self,vocab_path):
        vocab_path = os.path.join(vocab_path)
        with open(vocab_path, 'rb') as fr:
            word2id = pickle.load(fr)
        print('vocab_size:', len(word2id))
        return word2id

    '''加载词向量'''
    def load_wordvec(self,vocabs,word2vecpath = None):
        wordvec = None
        try:
            if word2vecpath == None:
                return wordvec
            else:
                import gensim
                wordvecmodel = gensim.models.Word2Vec.load(word2vecpath) # 读取词向量
                wordvec = {}
                for word in vocabs:
                    if word in wordvecmodel.wv.index2word:
                        wordvec[word] = wordvecmodel.wv[word]
        except:
            wordvec = None
        return wordvec

    # ==================================================================================================================
    # 数据处理
    # ==================================================================================================================

    def get_labels(self):
        id2tag = {}
        tag2id = {}
        for k,label in enumerate(self.labels):
            tag2id[label] = k+1
            id2tag[k+1] = label
        return tag2id,id2tag

    # ==================================================================================================================
    # ==================================================================================================================
    def train(self,model,x_train,y_train,x_val=None,y_val=None,checkpoint_flag = True):
        if checkpoint_flag == True:
            timestamp = str(int(time.time()))
            modelpath = os.path.join(self.modeldir,timestamp+'/checkpoint')
            if os.path.exists(modelpath) == False:
                os.makedirs(modelpath)
            filepath = modelpath+'/m-epoch-{epoch:02d}-val_acc-{val_acc:.2f}.h5'
            checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',mode='auto' ,save_best_only='True')
            callback_list = [checkpoint]
            if x_val is  None:
                history = model.fit(x_train,y_train,batch_size = self.batch_size,epochs=self.epochs, validation_split=0.3, verbose=1,callbacks=callback_list)
            else:
                history = model.fit(x_train,y_train,batch_size = self.batch_size,epochs=self.epochs,validation_data = (x_val,y_val),verbose=1,callbacks=callback_list)
        else:
            if x_val is  None:
                history = model.fit(x_train,y_train,batch_size = self.batch_size,epochs=self.epochs, validation_split=0.3, verbose=1)
            else:
                history = model.fit(x_train,y_train,batch_size = self.batch_size,epochs=self.epochs,validation_data = (x_val,y_val),verbose=1)
        model.save(os.path.join(self.modeldir,'hdcnn.m'))
        model.save_weights(os.path.join(self.modeldir,'hdcnn.m'))
        return history

    # ==================================================================================================================
    # ==================================================================================================================
    def doc_predict(self,x,modelpath):
        model = load_model(modelpath)
        model.summary()
        y_prob = model.predict(np.array(x))
        y_predict = np.argmax(y_prob,axis=2)
        y_label = [[self.id2tag[w] for w in s if w!=0] for s in y_predict]
        return y_label

if __name__ == '__main__':
    pass
    hlstm = HLSTM()
    model = hlstm.createmodel()