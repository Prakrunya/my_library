#IMPORTS

import pandas as pd
import numpy as np
import sys
import os

os.system('python -m spacy download en_core_web_md')
import en_core_web_md
nlp = en_core_web_md.load()
def nlp_test(s):
      return nlp(s)
  
my_github_name = 'uo-puddles'
my_library_name = 'uo_puddles'
clone_url = f'git clone https://github.com/{my_github_name}/{my_library_name}.git'  #create the url to get the library
os.system(clone_url) # Cloning

import uo_puddles.uo_puddles as up

#KKN

def average(number_list):
  summation = sum(number_list)
  average = summation/len(number_list)
  return average

def compute_sigma(number_list):
  diff_squared_list = [] 
  n = len(number_list)
  for i in range(n):
    item = number_list[i]
    summation = sum(number_list)
    mean = summation/len(number_list)
    sqdiff = (mean - item)**2
    diff_squared_list += [sqdiff]
  variance = sum(diff_squared_list)/n
  compute_sigma = variance**.5
  return compute_sigma

def compute_highwall(number_list):
  diff_squared_list = [] 
  n = len(number_list)
  for i in range(n):
    item = number_list[i]
    summation = sum(number_list)
    mean = summation/len(number_list)
    sqdiff = (mean - item)**2
    diff_squared_list += [sqdiff]
  variance = sum(diff_squared_list)/n
  sigma = variance**.5
  sigma3 = 3 * sigma
  mean = sum(number_list)/n
  high_wall = mean + sigma3
  return high_wall

def euclidean_distance(number_list1,number_list2):
  diff_squared_list = [] 
  n = len(number_list1)
  for i in range(n):
    item1 = int(number_list1[i])
    item2 = int(number_list2[i])
    sqdiff = (item1 - item2)**2
    diff_squared_list += [sqdiff]
  summation = sum(diff_squared_list)
  euclidean_distance = summation**.5
  return euclidean_distance

def knn(table, target_list, k):
  distance_record = []  
  n = len(table)
  for i in range(n):
    row = table.loc[i].to_list()
    number_list = row[:-1]
    choice = row[-1]  #last thing in list
    d = euclidean_distance(target_number_list, number_list)
    pair = [d, choice]
    distance_record += [pair]
  sorted_results = sorted(distance_record)
  return sorted_results[:k]

#NAIVE BAYES

def process_bio(bio):
  doc = nlp(bio)
  good_words = []
  for i in range(len(doc)):
    token = doc[i]
    if token.is_alpha and not token.is_oov and not token.is_stop:
      good_words += [token.text]
  return good_words

def class_probability(training_table, a_class):
  class_list = training_table['Class'].to_list()  #the Class column as a list
  class_count = class_list.count(a_class)
  return class_count/len(class_list)

def word_by_class_probability(training_table, word_bag, word, a_class, laplace=1):
  class_list = training_table['Class'].to_list()
  d = len(set(class_list))
  class_count = class_list.count(a_class)  #number of bios of a_class
  word_count = word_bag.loc[word, a_class] if word in word_bag.index else 0 #bios of a_class that used the word
  return (word_count+laplace)/(class_count + d*laplace)

def naive_bayes(training_table, word_bag, bio, a_class):
  good_words = process_bio(bio)
  n = len(good_words)
  numerator_list = [class_probability(training_table, a_class)]  #start if off with P(O)
  for i in range(n):
    word = good_words[i]
    word_class = word_by_class_probability(training_table, word_bag, word, a_class)
    numerator_list += [word_class]
  numerator = 0
  for number in numerator_list:
    numerator += math.log(number)
  return numerator

def all_bayes(training_table, word_bag, bio):
  all_classes = word_bag.columns.to_list()  #does not include word column because it is index
  results = []
  for i in range(len(all_classes)):
    c = all_classes[i]
    result = naive_bayes(training_table, word_bag, bio, c)
    results += [[result,c]]
  return sorted(results, reverse=True)

def predictions(test_table, training_table, word, target):
  test_results = []
  n = len(test_table)
  for i in range(n):
    test_bio = test_table.loc[i, word].lower()
    results = all_bayes(training_table, word_bag, test_bio)
    test_results += [results]
  predictions = []
  for i in range(n):
    r = test_results[i]
    first_pair = r[0]
    prediction = first_pair[1]
    predictions += [prediction]
  n = len(predictions)
  correct = 0
  actuals = test_table[target].to_list()
  for i in range(n):
    if predictions[i]==actuals[i]:
      correct += 1
  return correct/n

#Data Workup

def data_import(url,file_name):
  file_name = pd.read_csv(url)
  return file_name.head()

def table_divide(table):
  table = table.sample(frac=1.0, random_state=1)
  length = round(len(table)/3)
  test_table = table[:length].reset_index(drop=True)
  training_table = table[table:].reset_index(drop=True)
  return test_table.head()
