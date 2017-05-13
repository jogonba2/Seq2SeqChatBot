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


class Evaluation:

	@staticmethod
	def word_error_rate(x_train_ini, y_train_ini, tokenizer, max_word_index):
		bad_words = 0
		for i in range(len(x_train_ini)):
			x = x_train_ini[i]
			x_repr = to_categorical(tok.texts_to_sequences([x])[0], max_word_index+1)
			y_pred = decode(x_repr, keras_model, reverse_index, max_word_index)+" EOF"
			y_pred = y_pred.split()
			y_true = y_train_ini[i].split()
			w = (float(wer(y_true, y_pred)[0:-1])/100.0)*len(y_true)
			bad_words += w
		return w / len(x_train_ini)
