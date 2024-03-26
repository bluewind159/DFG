from bs4 import BeautifulSoup
import string
import os
import sys
import os.path
from os.path import join
from random import randint
from shutil import copyfile
import operator
from random import shuffle
import pickle
import re

def multi_label_atomic(directory, level_dict):
    """
    Loads labels and blurbs of dataset
    """
    data = []
    soup = BeautifulSoup(open(join(directory), 'rt', encoding='utf-8').read(), "html.parser")
    for book in soup.findAll('book'):
        categories = set([])
        book_soup = BeautifulSoup(str(book), "html.parser")
        for t in book_soup.findAll('topics'):
            s1 = BeautifulSoup(str(t), "html.parser")
            structure = ['d0', 'd1', 'd2', 'd3']
            for level in structure:
                for t1 in s1.findAll(level):
                    level_dict[level].add(str(t1.string))
                    categories.add(str(t1.string))
        data.append((str(book_soup.find("body").string), categories))
    return data, level_dict
    
#train_data = multi_label_atomic('./BlurbGenreCollection_EN_train.txt')
#eval_data  = multi_label_atomic('./BlurbGenreCollection_EN_dev.txt')
#test_data  = multi_label_atomic('./BlurbGenreCollection_EN_test.txt')
#root=list(set(root))
#print(root)