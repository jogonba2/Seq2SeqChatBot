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

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import numpy as np
np.random.seed(13)

class Utils:
	
	@staticmethod
	def bucketing(x, y, sample_weights):
	
		buckets_x = {}
		buckets_y = {}
		buckets_w = {}
		
		for i in range(len(x)):
			act_len = len(x[i])
			if act_len not in buckets_x:
				buckets_x[act_len] = [x[i]]
				buckets_y[act_len] = [y[i]]
				buckets_w[act_len] = [sample_weights[i]]
				
			else: 
				buckets_x[act_len].append(x[i])
				buckets_y[act_len].append(y[i])
				buckets_w[act_len].append(sample_weights[i])
				
		del x, y, sample_weights
		
		for bucket in buckets_x:
			buckets_x[bucket] = np.array(buckets_x[bucket])
			buckets_y[bucket] = np.array(buckets_y[bucket])
			buckets_w[bucket] = np.array(buckets_w[bucket])
			
		return buckets_x, buckets_y, buckets_w

	@staticmethod
	def get_samples_weight(x, MAX_LEN_OUTPUT):
		samples_weight = []
		for i in range(len(x)):
			
			if MAX_LEN_OUTPUT<len(x[i]):
				weight_one  = np.ones(MAX_LEN_OUTPUT)
				samples_weight.append(weight_one)
			else:
				weight_one = np.ones(len(x[i]))
				weight_zero = np.zeros(MAX_LEN_OUTPUT-len(x[i]))
				samples_weight.append(np.concatenate((weight_one, weight_zero), axis=0))
		return samples_weight

	@staticmethod
	def load_samples(x_file, y_file, input_embeddings, max_sents):
		f_x = open(x_file, encoding="utf8")
		f_y = open(y_file, "r", encoding="utf8")
		if input_embeddings: x_train = [xi.lower()+" EOF" for xi in f_x.read().split("\n")][:max_sents]
		else:                x_train = [xi.lower()+" EOF" for xi in f_x.read().split("\n")][:max_sents]
		y_train = [yi.lower()+" EOF"  for yi in f_y.read().split("\n")][:max_sents]
		
		return x_train, y_train

	@staticmethod
	def get_max_word_index(tokenizer):
		return tokenizer.word_index[max(tokenizer.word_index.keys(), key=(lambda k: tokenizer.word_index[k]))]
		
	@staticmethod
	def get_reverse_index(tokenizer):
		reverse_index = {}
		for word in tokenizer.word_index:
			reverse_index[tokenizer.word_index[word]] = word 
		return reverse_index

	@staticmethod
	def as_sequence(sequences, w2v_model, dim_embeddings, eof=False):
		seqs = []
		for i in range(len(sequences)):
			seqs.append([])
			cnt = 0
			for j in range(len(sequences[i])): 
				if sequences[i][j] in w2v_model: 
					seqs[-1].append(w2v_model[sequences[i][j]])
					cnt += 1
				else:
					seqs[-1].append(np.zeros(dim_embeddings)) # UNK #
			if eof: seqs[-1].append(np.zeros(dim_embeddings)+0.025) # EOF #

		return np.array(seqs)

	@staticmethod
	def one_hot_one_hot_representation(x_train, y_train):
		tok = Tokenizer()
		tok.fit_on_texts(x_train+y_train)
		max_word_index = Utils.get_max_word_index(tok)
		reverse_index  = Utils.get_reverse_index(tok)
		x_train = tok.texts_to_sequences(x_train)
		y_train = tok.texts_to_sequences(y_train)
		return x_train, y_train, max_word_index, reverse_index, tok
		
	@staticmethod
	def embedding_one_hot_representation(x_train, y_train, dim_embeddings, w2v_file):
		x_train_ini = x_train[:]
		x_train     = [text_to_word_sequence(x_train[i]) for i in range(len(x_train))]
		#w2v_model = Word2Vec.load(w2v_file)
		w2v_model = Word2Vec.load_word2vec_format(w2v_file, binary=True)
		#w2v_model = Word2Vec(x_train+y_train, size=dim_embeddings, window=5, min_count=0)
		x_train   = Utils.as_sequence(x_train, w2v_model, dim_embeddings, eof=False)
		tok = Tokenizer()
		tok.fit_on_texts(x_train_ini+y_train)
		max_word_index = Utils.get_max_word_index(tok)
		reverse_index  = Utils.get_reverse_index(tok)
		y_train = tok.texts_to_sequences(y_train)
		return x_train, y_train, max_word_index, reverse_index, tok, w2v_model
		
	@staticmethod
	def embedding_embedding_representation(x_train, y_train, dim_embeddings, w2v_file):
		x_train     = [text_to_word_sequence(x_train[i]) for i in range(len(x_train))]
		y_train     = [text_to_word_sequence(y_train[i]) for i in range(len(y_train))]
		#w2v_model = Word2Vec.load(w2v_file)
		w2v_model = Word2Vec.load_word2vec_format(w2v_file, binary=True)
		#w2v_model = Word2Vec(x_train+y_train, size=dim_embeddings, window=5, min_count=0)
		x_train   = Utils.as_sequence(x_train, w2v_model, dim_embeddings, eof=False)
		y_train   = Utils.as_sequence(y_train, w2v_model, dim_embeddings, eof=False)
		return x_train, y_train, w2v_model

	@staticmethod
	def padding_output_one_hot(y, max_word_index, MAX_LEN_OUTPUT):
		y = pad_sequences(y, MAX_LEN_OUTPUT, padding='post', truncating='post', dtype="float64", value=max_word_index+1)
		return y

	@staticmethod
	def padding_output_embedding(y, MAX_LEN_OUTPUT):
		y = pad_sequences(y, MAX_LEN_OUTPUT, padding='post', truncating='post', dtype="float64", value=-0.037)
		return y

	@staticmethod		
	def samples_to_categorical(x_train, y_train, max_word_index):
		for i in range(len(x_train)):
			x_train[i] = to_categorical(x_train[i], max_word_index+2) 
			y_train[i] = to_categorical(y_train[i], max_word_index+2)
			
		x_train = np.array(x_train)
		y_train = np.array(y_train)

		return x_train, y_train

	@staticmethod
	def output_to_categorical(y_train, max_word_index):
		for i in range(len(y_train)):
			y_train[i] = to_categorical(y_train[i], max_word_index+2)
			
		y_train = np.array(y_train)

		return y_train
		
	@staticmethod
	def reverse_input(x): return x[::-1]
