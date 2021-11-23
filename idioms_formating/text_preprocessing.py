from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import re


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    print(matches)
    print(m.group(0) for m in matches)
    return [m.group(0) for m in matches]


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
    # group_of_tokens = [camel_case_split(token) for token in tokens]
    # tokens = [token for group in group_of_tokens for token in group]
    # remove stop words
    # tokens = [x for x in tokens if x not in stopwords.words('english')]
    # stemming
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(t) for t in tokens]

    return tokens
