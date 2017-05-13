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

import sys
sys.path.insert(0, '../Seq2Seq')
from Utils import Utils as u
from S2S import Sequence2Sequence as s2s
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
import telebot

def app_embedding_one_hot(x_train_file, y_train_file, reverse, dim_embeddings, w2v_file, MAX_LEN_OUTPUT, MAX_LENGTH_SENTS):
	input_embeddings = True
	output_embeddings = False
	x_train, y_train = u.load_samples(x_train_file, y_train_file, input_embeddings, MAX_LENGTH_SENTS) # x_train_cleverbot.txt #
	x_train, y_train, max_word_index, reverse_index, tokenizer, w2v_model = u.embedding_one_hot_representation(x_train, 
																											 y_train, 
																											 dim_embeddings,
																											 w2v_file)	 
	sample_weights = u.get_samples_weight(y_train, MAX_LEN_OUTPUT)																										 
	y_train = u.padding_output_one_hot(y_train, max_word_index, MAX_LEN_OUTPUT).tolist()
	y_train = u.output_to_categorical(y_train, max_word_index)
	if reverse: x_train = u.reverse_input(x_train)	
	buckets_x, buckets_y, buckets_w = u.bucketing(x_train, y_train, sample_weights)
	seq2seq_model = s2s.get_seq2seq_model_one_hot(dim_embeddings, max_word_index+2, MAX_LEN_OUTPUT)
	for i in range(1):
		# Train #
		print(i)
		for j in buckets_x.keys():
			hist = seq2seq_model.fit(buckets_x[j], buckets_y[j], sample_weight=buckets_w[j], nb_epoch=1, verbose=True)
	seq2seq_model.save("bot_9_en.h5")
	seq2seq_model = load_model("bot_9_en.h5")
	print("All loaded.")
	TOKEN = 'TOKEN'
	cbot = telebot.TeleBot(TOKEN)
	def listener(*messages):
		for m in messages:
			m = m[0]
			cid = m.chat.id
			if m.content_type=="text":
				text = m.text.lower().strip()+" EOF"
				test_sample = text_to_word_sequence(text)
				test_sample = u.as_sequence([test_sample], w2v_model, dim_embeddings, eof=True)[0]
				if reverse: test_sample = u.reverse_input(test_sample)
				res = s2s.decode_one_hot(test_sample, seq2seq_model, reverse_index, max_word_index)
				cbot.send_message(cid, res)
			else:
				cbot.send_message(cid, "That content is not allowed")
	cbot.set_update_listener(listener)
	cbot.polling()
	print("Processing messages..")
	while True: pass

if __name__ == '__main__': app_embedding_one_hot("x_train_cleverbot.txt", "y_train_cleverbot.txt", 
												 False, 20, "Word2Vec.model", 50, 9999)
