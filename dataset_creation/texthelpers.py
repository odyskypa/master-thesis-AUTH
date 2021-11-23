import os
import re
import sys
import csv
import string
from nltk.tokenize import word_tokenize, RegexpTokenizer
from properties import data_folder_path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]

def process_text(text, removestopwords = True, splitcamelcase = True):
    text = text.translate({ord(c): ' ' for c in string.punctuation})
    tokens = word_tokenize(text)
    if splitcamelcase:
        tokens = [camel_case_split(t) for t in tokens]
        tokens = [item for sublist in tokens for item in sublist]

    if removestopwords:
        tokens = [t for t in tokens if t not in stopwords.words('english')]

    text = ' '.join(tokens)
    
    return text

def preprocessor(text):
    """
    Apply several preprocessing steps: tokenization, split camel case, remove stop words, stemming

    :param text
    :return: tokens
    """

    # tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # print(tokens)
    # tokens = word_tokenize(text)
    # print(tokens)

    # print(tokens)
    # split camel case words
    #group_of_tokens = [camel_case_split(token) for token in tokens]
    #tokens = [token for group in group_of_tokens for token in group]
    # remove stop words
    # tokens = [x for x in tokens if x not in stopwords.words('english')]
    # stemming
    #stemmer = PorterStemmer()
    #tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def set_max_int_for_csv():
    #----The csv file might contain very huge fields, therefore increase the field_size_limit----
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
