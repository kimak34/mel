from mynn.layers.dense import dense
from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.activations import relu

import mygrad as mg
import numpy as np

import os
import pickle

import string

from cogworks_data.language import get_data_path
from gensim.models import KeyedVectors
glove = KeyedVectors.load_word2vec_format(get_data_path("glove.6B.200d.txt.w2v"), binary=False)

class Mel:
    def __init__(self, dim_input, dim_recurrent, dim_output, seq_length, id_to_label=None, responses=None):
        self.Wx = dense(dim_input, dim_recurrent, weight_initializer=glorot_normal)
        self.Wh = dense(dim_recurrent, dim_recurrent, weight_initializer=glorot_normal, bias=False)
        self.Wy = dense(seq_length*dim_recurrent, dim_output, weight_initializer=glorot_normal)
        self.seq_length = seq_length
        
        self.id_to_label = id_to_label
        self.responses = responses
    
    def __call__(self, x, h=None):
        h_t = np.zeros((1, x.shape[1], self.Wh.weight.shape[0]), dtype=np.float32) if h is None else h
        h = []
        
        for x_t in x:
            h_t = relu(self.Wx(x_t[np.newaxis]) + self.Wh(h_t))
            h.append(h_t)
        
        h = mg.concatenate(h, axis=0)
        h = mg.reshape(h, [len(h), -1])
        
        return self.Wy(h), h
    
    @property
    def parameters(self):
        return self.Wx.parameters + self.Wh.parameters + self.Wy.parameters
    
    def preprocess(self, text):
        for p in string.punctuation:
            text = text.replace(p, "")
        text = text.lower().strip()
        embedding = self.embed(text)

        while len(embedding) < self.seq_length:
            embedding = np.vstack((embedding, np.zeros(200).reshape(1,200)))
        return embedding

    def embed(self, text):
        embeddings = []
        for word in text.split(" "):
            try:
                embeddings.append(glove[word])
            except KeyError:
                embeddings.append(np.zeros(200))
        return embeddings
    
    def query(self, prompt):
        tag = self.id_to_label[np.argmax(self(np.expand_dims(self.preprocess(prompt), 0))[0][0])]
        return np.random.choice(self.responses[tag]), tag
    
    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for i, parameter in enumerate(self.parameters):
            np.save(os.path.join(dir_path, "parameter{}.npy".format(i)), parameter.data)
        with open(os.path.join(dir_path, "id_to_label"), mode='wb') as file:
            pickle.dump(self.id_to_label, file)
        with open(os.path.join(dir_path, "responses"), mode='wb') as file:
            pickle.dump(self.responses, file)

    def load(self, dir_path):
        self.Wx.weight = np.load(os.path.join(dir_path, "parameter0.npy"))
        self.Wx.bias = np.load(os.path.join(dir_path, "parameter1.npy"))
        self.Wh.weight = np.load(os.path.join(dir_path, "parameter2.npy"))
        self.Wy.weight = np.load(os.path.join(dir_path, "parameter3.npy"))
        self.Wy.bias = np.load(os.path.join(dir_path, "parameter4.npy"))
        with open(os.path.join(dir_path, "id_to_label"), mode='rb') as data:
            self.id_to_label = pickle.load(data)
        with open(os.path.join(dir_path, "responses"), mode='rb') as data:
            self.responses = pickle.load(data)
        