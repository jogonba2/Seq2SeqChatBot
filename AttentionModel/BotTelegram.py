#!/usr/bin/env python
# -*- coding: utf-8 -*-
import telebot
import os
import random
import html
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def _get_response(req):
    id_sample = str(random.randint(10000, 1000000))
    with open(id_sample, "w") as fw: fw.write(req)
    tokenize = "./tokenizer.perl < " + id_sample + " > " + id_sample + ".tok"
    os.system(tokenize + " 2> /dev/null")
    infer = """
python3 -m bin.infer \\
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \\
  --model_dir ./S2SModels/  \\
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
        source_files:
          - ./""" + id_sample + """ " \\
    > ./""" + id_sample + """.preds """
    os.system(infer + "2> /dev/null")
    res = "UNK"
    with open(id_sample+".preds", "r") as fr: res = fr.read()
    os.system("rm "+id_sample)
    os.system("rm "+id_sample+".tok")
    os.system("rm "+id_sample+".preds")
    return res

def main():
    TOKEN = 'TOKEN'
    cbot = telebot.TeleBot(TOKEN)
    def listener(*messages):
        for m in messages:
            m = m[0]
            cid = m.chat.id
            if m.content_type=="text":
                text = m.text
                res = html.unescape(_get_response(text))
                cbot.send_message(cid, res)

    print("Processing messages...")
    while True:
        try:
            cbot.set_update_listener(listener)
            cbot.polling(none_stop=True)
        except:
            print("Reset.")
            cbot.set_update_listener(listener)
            cbot.polling(none_stop=True)

if __name__ == "__main__":
    with suppress_stdout():
        main()
