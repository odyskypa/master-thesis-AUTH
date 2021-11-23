from sklearn.feature_extraction.text import CountVectorizer
from dataset_creation.texthelpers import preprocessor
import string
import re


def idiom_general_form(cluster_codes, centroid):
    """
    :param cluster_codes
    :param centroid
    :return: abstract idiom
    """

    vectorizer = CountVectorizer(tokenizer=preprocessor, lowercase=False, binary=True)

    bag_of_words = vectorizer.fit_transform(cluster_codes)
    #print(len(cluster_codes))
    print(bag_of_words)
    # find the most common words in the cluster
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    # print(words_freq)
    common_words = [words_freq[j][0] for j in range(len(words_freq)) if words_freq[j][1]/len(cluster_codes) > 0.5]
    print(common_words)
    centroid_tokens = preprocessor(centroid)

    general_centroid = centroid
    print(general_centroid)
    const_list = ['Do', 'While', 'If', 'return', 'Switch', 'case', 'null', 'double', 'int', 'char', 'String', 'break', 'not','open', 'get', 'set', 'close']

    centroid_tokens = list(dict.fromkeys(centroid_tokens))

    variables_counter = 1
    methods_counter = 1
    object_counter = 1
    constant_counter = 1
    list_counter = 1
    class_counter = 1

    # print(centroid_tokens)
    for word in centroid_tokens:

        if word not in common_words and word not in const_list:

            index = centroid.index(word)
            # print(index)
            next_character = centroid[index+len(word)]

            if len(word) == 1:
                
                if next_character == '.':
                    general_centroid = general_centroid.replace(word, "$(object" + str(object_counter) + ")")
                    object_counter += 1
                
                general_centroid = re.sub(r'[(]%s[)]' % re.escape(word), "$(simpleVariable" + str(variables_counter) + ")",
                                          general_centroid)

                general_centroid = re.sub(r'[ ]%s[ ]' % re.escape(word), " $(simpleVariable" + str(variables_counter) + ") ",
                                          general_centroid)

                # print(a.index())
                # previous_character = general_centroid[index-1]
                # general_centroid = general_centroid.replace(previous_character+word, previous_character +
                                                            # "$(simpleVariable" + str(variables_counter) + ")")
                variables_counter += 1
                continue

            if next_character == '(':
                general_centroid = general_centroid.replace(word, "$(method" + str(methods_counter) + ")")
                methods_counter += 1
            elif next_character == '.':
                general_centroid = general_centroid.replace(word, "$(object" + str(object_counter) + ")")
                object_counter += 1
            elif word.isupper():
                general_centroid = general_centroid.replace(word, "$(CONSTANT_VALUE" + str(constant_counter) + ")")
                constant_counter += 1
            elif centroid[index-2] == ":":
                general_centroid = re.sub(r'\b%s\b' % re.escape(word),
                                          "$(iterList" + str(list_counter) + ")",
                                          general_centroid)

                # general_centroid = general_centroid.replace(word, "$(list" + str(constant_counter) + ")")
                list_counter += 1
            elif word[0].isupper():
                general_centroid = re.sub(r'\b%s\b' % re.escape(word),
                                          "$(Class" + str(class_counter) + ")",
                                          general_centroid)
                class_counter += 1

            else:
                general_centroid = re.sub(r'\b%s\b' % re.escape(word), "$(simpleVariable" + str(variables_counter) + ")",
                                          general_centroid)

                #general_centroid = general_centroid.replace(word, "$(simpleVariable" + str(variables_counter) + ")")
                #general_centroid = general_centroid.replace(word.capitalize(), "$(simpleVariable" + str(variables_counter) + ")")
                variables_counter += 1

    #print(general_centroid)

    return general_centroid
