# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:58:13 2020

@author: robbf
"""

#Import all dependencies
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import gensim
import nltk
nltk.download('stopwords')

import datetime

import tensorflow_hub as hub
import tensorflow as tf
import zipfile
import PyPDF2 as pdf
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from io import BytesIO, StringIO

import re

from flask import (
        Flask, url_for
        )

from os.path import join, dirname, realpath

import os

from google.cloud import storage

KEEP = False
KEY_WORDS = ['responsibility','responsibilities','duties','duty']
STOP_WORDS = ['page','position']
raw_task_input = []
raw_title_input = []

def from_colon_on(sentence):
  try:
    colon_index = sentence.index(":")
    return sentence[colon_index:]
  except:
    return sentence

def from_hyphon_on(sentence):
  try:
    hyphon_index = sentence.index("-")
    return sentence[hyphon_index:]
  except:
    return sentence

def pdf_to_text(pdf_file,title):
    manager = PDFResourceManager()
    retstr = BytesIO()
    layout = LAParams(all_texts=True)
    device = TextConverter(manager,retstr,laparams=layout)
    interpreter =  PDFPageInterpreter(manager,device)
    for page in PDFPage.get_pages(pdf_file,check_extractable=True):
      interpreter.process_page(page)
    text = retstr.getvalue()
    device.close()
    retstr.close()
    PUNCTUATION = [',','/n','(',')',':']
    text = str(text,'utf-8')
    
    text = [word for word in text.split() if word not in PUNCTUATION]
    text = [''.join([letter for letter in word if letter not in PUNCTUATION]) for word in text]
    text = [''.join(['.' if letter in [';','-'] else letter for letter in word]) for word in text]
    text = ['example' if word == 'e.g.' else word for word in text]
    
    text = ' '.join(text)
    text = text.split('.')
    STOP_WORDS = ['description','page','description','jobcode','job']
    text = [line for line in text if len(line.split()) > 6]
    text = [line for line in text if len([word.lower() for word in line.split() if word.lower() in STOP_WORDS]) == 0]
    for task in text:
      yield task, title

def unzip_client_JD_file(zip_file_):
  with zipfile.ZipFile(zip_file_) as jd_zip:
    X = 1
    for jd in jd_zip.infolist():
      title = jd.filename
      jd = jd_zip.extract(jd)
      pdfreader = pdf.PdfFileReader(jd)
      X += 1
      for page in range(0,pdfreader.numPages):
        pdfPageObj = pdfreader.getPage(page)
        for sentence in pdfPageObj.extractText().split("."):
          for task in sentence.split("-"):
            for sub_task in task.split(":"):
              for sub_3_task in sub_task.split(";"):
                for sub_4_task in sub_3_task.split("."):
                  yield sub_4_task, title

def gather_tasks(zip_file_, zip_=True, title=None):
    if zip_ == True:
        for sentence, title in unzip_client_JD_file(zip_file_):
          
          key_word_len = len([word for word in sentence.split() if word.lower() in KEY_WORDS])
          word_len = len([word for word in sentence.split()])
          stop_word_len = len([word for word in sentence.split() if word.lower() in STOP_WORDS])
        
          if key_word_len >= 1:
            KEEP = True
            pass
          elif stop_word_len >= 1:
            KEEP = False
            pass
          elif KEEP == True and word_len >= 6:
            raw_task_input.append(sentence)
            raw_title_input.append(title)
          else:
            pass
        
        return raw_task_input, raw_title_input
    
    elif zip_==False:
        for sentence in yield_tasks(zip_file_):
          
          key_word_len = len([word for word in sentence.split() if word.lower() in KEY_WORDS])
          word_len = len([word for word in sentence.split()])
          stop_word_len = len([word for word in sentence.split() if word.lower() in STOP_WORDS])
        
          if word_len >= 6:
              sentence = [word for word in sentence.split() if "/n" not in word]
              sentence = " ".join(sentence)
              raw_task_input.append(sentence)
              raw_title_input.append(title)
          else:
              pass
        
        return raw_task_input, raw_title_input

def yield_tasks(pdf_file):
    pdfreader = pdf.PdfFileReader(pdf_file)
    for page in range(0,pdfreader.numPages):
        pdfPageObj = pdfreader.getPage(page)
        for sentence in pdfPageObj.extractText().split("."):
            for task in sentence.split("-"):
                yield task

def init_csv():
    storage_client = storage.Client.from_service_account_json(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    
    #Identify the data to pull for the Tensorflow model
    bucket = storage.Client().bucket('granularity-tasks1')
    blob = bucket.blob('data/onet_train.csv')
    blob.download_to_filename('onet_train.csv')
    
    df = pd.read_csv('onet_train.csv') #Read the dataframe
    
    #Initiate the new dataframe with the columns we need
    task_df = pd.DataFrame()
    task_df['task'] = df['Task']
    task_df['title'] = df['Title']
    task_df['clusters'] = df['IWA_clusterid']
    
    return task_df 

def fit_tokenizer(text,tokenizer):
  tokenizer.fit_on_texts(text)
  return tokenizer

def encode_text(text,tokenizer,length):
  encoded = tokenizer.texts_to_sequences(text)
  padded = tf.keras.preprocessing.sequence.pad_sequences(encoded,
                                                         maxlen=length,
                                                         padding='post')
  return padded

def prepare_JD_tasks(array,input_='task'):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=8000+1)

  task_df = init_csv()
  tokenizer = fit_tokenizer(task_df['task'],tokenizer)

  length = max([len(s.split()) for s in task_df['task']])
  
  encoded_JD_tasks = encode_text(array,tokenizer,length)

  return encoded_JD_tasks

def load_model():
    storage_client = storage.Client.from_service_account_json(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    
    #Identify the data to pull for the Tensorflow model
    bucket = storage.Client().bucket('granularity-tasks1')
    blob = bucket.blob('models/granularity_model_3CNN_MT_ATT_v1.h5')
    blob.download_to_filename('granularity_model_3CNN_MT_ATT_v1.h5')
    
    model = tf.keras.models.load_model("granularity_model_3CNN_MT_ATT_v1.h5")
    return model

def IWA_from_id(array,df,ONET_code='IWA'):
    return_list = []
    for x in array:
        IWA_id = df[df[ONET_code+'_clusterid'] == x][ONET_code+'_cluster'].iloc[0]
        return_list.append(IWA_id)
    return return_list

def get_IWA_prob(dataframe,predictions):
    return_list = []
    for x in dataframe.index:
        return_list.append(predictions[x].max())
    return return_list

def get_IWA_title(id_, IWA_df):
    return IWA_df[IWA_df['IWA ID'] == id_]['IWA Title'].iloc[0]

def get_GWA_title(id_,GWA_df):
    return GWA_df[GWA_df['Element ID'] == id_]['Element Name'].iloc[0]

def determine_predictions(predictions,raw_task_input,raw_title_input):
    
    storage_client = storage.Client.from_service_account_json(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    
    #Identify the data to pull for the Tensorflow model
    bucket = storage.Client().bucket('granularity-tasks1')
    blob = bucket.blob('data/IWA Reference.xlsx')
    blob.download_to_filename('IWA_Reference.xlsx')

    blob = bucket.blob('data/onet_train2.csv')
    blob.download_to_filename('onet_train2.csv')
    
    blob = bucket.blob('data/Work Activities.xlsx')
    blob.download_to_filename('Work_Activities.xlsx')
    
    ONET_df = pd.read_csv('onet_train2.csv') #Read the dataframe
    IWA_df = pd.read_excel("IWA_Reference.xlsx")
    GWA_df = pd.read_excel("Work_Activities.xlsx")
    
    pred_IWAclusters = predictions[0].argmax(-1)
    pred_GWAclusters = predictions[1].argmax(-1)
    
    transfer_df = pd.DataFrame()
    transfer_df['task'] = raw_task_input
    transfer_df['IWA_clusterid'] = pred_IWAclusters
    transfer_df['GWA_clusterid'] = pred_GWAclusters
    transfer_df['IWA_actual_id'] = IWA_from_id(pred_IWAclusters,ONET_df)
    transfer_df['GWA_actual_id'] = IWA_from_id(pred_GWAclusters,ONET_df,ONET_code='GWA')
    transfer_df['IWA_probability'] = get_IWA_prob(transfer_df,predictions[0])
    transfer_df['GWA_probability'] = get_IWA_prob(transfer_df,predictions[1])
    transfer_df['IWA_actual_title'] = transfer_df['IWA_actual_id'].apply(lambda x: get_IWA_title(x,IWA_df))
    transfer_df['GWA_actual_title'] = transfer_df['GWA_actual_id'].apply(lambda x: get_GWA_title(x,GWA_df))
    transfer_df['Position_title'] = raw_title_input
    
    return transfer_df
    

def run_predictions_zip(zip_file_,zip_=True,title=None):
    
    raw_task_input, raw_title_input = gather_tasks(zip_file_,zip_,title)
    
    encoded_tasks = prepare_JD_tasks(raw_task_input)
    encoded_titles = prepare_JD_tasks(raw_title_input)
    
    model = load_model()
    
    predictions = model.predict([encoded_tasks,encoded_tasks,encoded_titles])
    
    transfer_df = determine_predictions(predictions,raw_task_input,raw_title_input)
    
    return transfer_df
    
def run_predictions(raw_task_input,raw_title_input):
    
    encoded_tasks = prepare_JD_tasks(raw_task_input)
    encoded_titles = prepare_JD_tasks(raw_title_input)
    
    model = load_model()
    
    predictions = model.predict([encoded_tasks,encoded_tasks,encoded_titles])
    
    transfer_df = determine_predictions(predictions,raw_task_input,raw_title_input)
    
    return transfer_df
    
    
    