#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "José Ángel González"
__author2__ = "Pascual Andrés Carrasco Gómez"
__license__ = "GNU"
__version__ = "0.2"
__maintainer__ = "José Ángel González"
__maintainer2__ = "Pascual Andrés Carrasco Gómez"
__email__ = "jogonba2@dsic.upv.es"
__email2__ = "pascargo@dsic.upv.es"
__status__ = "Versión final"

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from scipy.spatial.distance import cosine
import numpy as np
np.random.seed(13)

class Sequence2Sequence:
	
	@staticmethod
	def decode_one_hot(input, keras_model, reverse_index, max_word_index): 
		predicted = keras_model.predict(np.array([input]))[0]
		res = []
		for i in range(len(predicted)):
			argmax = np.argmax(predicted[i])
			if argmax==max_word_index-1: 
				res.append("EOF") ; break
			elif argmax==max_word_index: pass
			else: 
				if reverse_index[argmax]!="eof" and reverse_index[argmax]!="EOF":
					res.append(reverse_index[argmax])
				else:
					res.append("EOF"); break
		return " ".join(res)
	
	@staticmethod
	def decode_embedding(input, keras_model, w2v_model, dim_embeddings):
		print(input)
		predicted = keras_model.predict(np.array([input]))[0]
		res = []
		pad = np.zeros(dim_embeddings)-0.037
		for i in range(len(predicted)):
			aux = w2v_model.most_similar([predicted[i]], [], topn=1)[0]
			if cosine(pad, predicted[i])<aux[1]: continue
			else: res.append(aux[0])
			if res[-1]=="eof" or res[-1]=="EOF": break
		return " ".join(res)
	
	@staticmethod
	def get_seq2seq_model_one_hot(input_size, output_size, MAX_LEN_OUTPUT):
		seq2seq_model = Sequential()
		seq2seq_model.add(GaussianNoise(0.15, input_shape=(None, input_size))) # El modelo original no tenia GN
		seq2seq_model.add(BatchNormalization()) # El modelo original no tenia BN
		seq2seq_model.add(LSTM(750, return_sequences=False, activation="relu")) # max_word_index+2 for one_hot, dim_embeddings for embedding
		seq2seq_model.add(RepeatVector(MAX_LEN_OUTPUT))
		seq2seq_model.add(GaussianNoise(0.15))
		seq2seq_model.add(BatchNormalization()) # El modelo original no tenia BN
		seq2seq_model.add(LSTM(950, return_sequences=True, activation="relu"))
		seq2seq_model.add(BatchNormalization()) # El modelo original no tenia BN
		seq2seq_model.add(TimeDistributed(Dense(output_size, activation="softmax"))) # max_word_index+2 for one_hot, dim_embeddings for embedding
		seq2seq_model.compile(optimizer='adadelta',
							sample_weight_mode="temporal",
							loss='categorical_crossentropy',
							metrics=['accuracy'])
		return seq2seq_model	
		
	@staticmethod	
	def get_seq2seq_model_embedding(input_size, output_size, MAX_LEN_OUTPUT):
		seq2seq_model = Sequential()
		seq2seq_model.add(BatchNormalization(input_shape=(None, input_size)))
		seq2seq_model.add(LSTM(256, return_sequences=False, activation="relu"))
		seq2seq_model.add(RepeatVector(MAX_LEN_OUTPUT))
		seq2seq_model.add(BatchNormalization())
		seq2seq_model.add(LSTM(512, return_sequences=True, activation="relu"))
		seq2seq_model.add(BatchNormalization())
		seq2seq_model.add(TimeDistributed(Dense(output_size, activation="linear"))) # max_word_index+2 for one_hot, dim_embeddings for embedding
		seq2seq_model.compile(optimizer='rmsprop',
								sample_weight_mode="temporal",
								loss='mse',
								metrics=['accuracy'])
		return seq2seq_model	
			
	# TODO #
	@staticmethod
	def get_attention_seq2seq_model(max_word_index, MAX_LEN_OUTPUT):
		
		# Encoder #
		input = Input(shape=(None, max_word_index+2))
		activations = LSTM(256, return_sequences=True, activation="relu")(input)
		
		# Atention #
		mask = TimeDistributed(Dense(1, activation="softmax"))(activations)
		flat = Flatten()(mask)
		activations = merge([activations, mask], mode="mul")
		activations = AveragePooling1D(pool_length=MAX_LEN_OUTPUT)(activation)
		activations = Flatten()(activations)
		
		# Decoder #
		seq2seq_model.add(RepeatVector(MAX_LEN_OUTPUT))
		seq2seq_model.add(LSTM(512, return_sequences=True, activation="relu"))
		seq2seq_model.add(TimeDistributed(Dense(max_word_index+2, activation="softmax")))
		seq2seq_model.compile(optimizer='rmsprop',
							sample_weight_mode="temporal",
							loss='categorical_crossentropy',
							metrics=['accuracy'])
		return seq2seq_model
	
