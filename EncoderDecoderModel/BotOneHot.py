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
from keras.models import load_model
import telebot


def load_new_samples(fx_name, fy_name):
	res_x, res_y = [], []
	fx = open(fx_name, "r")
	fy = open(fy_name, "r")
	data_x, data_y = fx.read().split("\n"), fy.read().split("\n")
	for i in range(len(data_x)):
		if data_x[i]!="" and data_y[i]!="":
			res_x.append(data_x[i])
			res_y.append(data_y[i])
	fx.close()
	fy.close()
	return res_x, res_y


def app_one_hot_one_hot(x_train_file, y_train_file, reverse, MAX_LEN_OUTPUT, MAX_LENGTH_SENTS):
	input_embeddings = False
	output_embeddings = False
	x_train, y_train = u.load_samples(x_train_file, y_train_file, input_embeddings, MAX_LENGTH_SENTS) # x_train_cleverbot.txt #
	x_train, y_train, max_word_index, reverse_index, tokenizer = u.one_hot_one_hot_representation(x_train, 
																								y_train)
	sample_weights = u.get_samples_weight(y_train, MAX_LEN_OUTPUT)
	y_train = u.padding_output_one_hot(y_train, max_word_index, MAX_LEN_OUTPUT).tolist()
	x_train, y_train = u.samples_to_categorical(x_train, y_train, max_word_index)	
	if reverse: x_train = u.reverse_input(x_train)
	buckets_x, buckets_y, buckets_w = u.bucketing(x_train, y_train, sample_weights)
	seq2seq_model = s2s.get_seq2seq_model_one_hot(max_word_index+2, max_word_index+2, MAX_LEN_OUTPUT)
	#for i in range(1):
		# Train #
	#	print(i)
	#	for j in buckets_x.keys():
	#		hist = seq2seq_model.fit(buckets_x[j], buckets_y[j], sample_weight=buckets_w[j], nb_epoch=1, verbose=True)
	#seq2seq_model.save("cleverbot_7_en.h5")
	seq2seq_model = load_model("cleverbot_7_en.h5")
	print("All loaded.")
	TOKEN = 'TOKEN'
	cbot = telebot.TeleBot(TOKEN)
	corrections = {}
	last_utterance = {}
	RETRAINING_SAMPLES = 2
	samples_x, samples_y = load_new_samples("new_samples_x.txt", "new_samples_y.txt")
	def listener(*messages):
		for m in messages:
			m = m[0]
			cid = m.chat.id
			if m.content_type=="text":
				text = m.text.strip().replace("\n","")
				if text=="❌" and cid in last_utterance:
					corrections[cid] = True
					cbot.send_message(cid, "Ok, then give me a valid example for your last utterance: "+last_utterance[cid])
				else:
					if cid in corrections and corrections[cid]:
						fx = open("new_samples_x.txt", "a")
						fy = open("new_samples_y.txt", "a")
						fx.write(last_utterance[cid]+"\n")
						fy.write(text+" EOF\n")
						fx.close()
						fy.close()
						del corrections[cid]
						cbot.send_message(cid, "Supervised sample: "+str(last_utterance[cid])+" - "+str(text))
						cbot.send_message(cid, "The new sample could be considered for re-training, thanks (heart)")
						samples_x.append(last_utterance[cid])
						samples_y.append(text)
					else:
						text = text+" EOF"
						last_utterance[cid] = text
						test_sample = to_categorical(tokenizer.texts_to_sequences([text])[0], max_word_index+2)
						if reverse: test_sample = u.reverse_input(test_sample)
						res = s2s.decode_one_hot(test_sample, seq2seq_model, reverse_index, max_word_index)
						cbot.send_message(cid, res)
			else:
				cbot.send_message(cid, "That content is not allowed")
				
		if len(samples_x)>=RETRAINING_SAMPLES:
			retrain_sample_x = tokenizer.texts_to_sequences(samples_x)
			retrain_sample_y = tokenizer.texts_to_sequences(samples_y)
			resample_weights = u.get_samples_weight(retrain_sample_y, MAX_LEN_OUTPUT)
			retrain_sample_y = u.padding_output_one_hot(retrain_sample_y, max_word_index, MAX_LEN_OUTPUT).tolist()
			retrain_sample_x, retrain_sample_y = u.samples_to_categorical(retrain_sample_x, retrain_sample_y, max_word_index)
			# Erase content of files #
			for i in range(len(retrain_sample_x)):
				if samples_x!="" and samples_y!="":
					# Reentrenar el modelo #
					print(retrain_sample_x[i].shape)
					print(retrain_sample_y[i].shape)
					print(resample_weights[i].shape)
					seq2seq_model.fit(np.array([retrain_sample_x[i]]), np.array([retrain_sample_y[i]]), 
							  nb_epoch=10, sample_weight=np.array([resample_weights[i]]))
					seq2seq_model.save("cleverbot_7_en.h5")
			with open("new_samples_x.txt", "w"): pass
			with open("new_samples_y.txt", "w"): pass
			for i in range(len(samples_x)):
				samples_x.pop(0)
				samples_y.pop(0)
	cbot.set_update_listener(listener)
	cbot.polling()
	print("Processing messages..")
	while True: pass

if __name__ == '__main__': app_one_hot_one_hot("x_train_cleverbot.txt", "y_train_cleverbot.txt", 
											   False, 20, 9999)
