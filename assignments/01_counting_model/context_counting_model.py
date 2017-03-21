# uncompyle6 version 2.9.10
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 |Anaconda 4.0.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
# [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
# Embedded file name: /home/niklas/ML-KA/tensorflow/tf_stanford/tf-stanford-tutorials/assignments/01_counting_model/context_counting_model.py
# Compiled at: 2017-03-21 17:53:08
# Size of source mod 2**32: 5332 bytes
""" Generate a co-occurence matrix  with tensorflow.
In this approach, co-occurence of words are counted and transformed to a
word embedding via SVD. This is different to context-predicting models
like in ../examples/04...
Many methods are inspired by the provided examples
All calculations are done using tensorflow.
"""
import tensorflow as tf
import zipfile
from collections import Counter
import numpy as np

class CooccurenceMatrix:
    """ Build a cooccurence matrix """

    def __init__(self, vocab_size, embed_size, skip_window):
        """ Init the model """
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.skip_window = skip_window

    def read_data(self, file_path):
        """
        Read data into a list of tokens
        There should be 17,005,207 tokens
        """
        with zipfile.ZipFile(file_path) as f:
            words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return words

    def build_vocab(self, words, vocab_size):
        """ Build vocabulary of VOCAB_SIZE most frequent words """
        dictionary = dict()
        count = [('UNK', -1)]
        count.extend(Counter(words).most_common(vocab_size - 1))
        index = 0
        export_file_name = ''.join(['vocab_', str(vocab_size), '.dsv'])
        with open(export_file_name, 'w+') as f:
            for word, _ in count:
                dictionary[word] = index
                if index < 1000:
                    f.write(word + '\n')
                index += 1

        index_dictionary = dict(list(zip(list(dictionary.values()), list(dictionary.keys()))))
        return (
         dictionary, index_dictionary)

    def convert_words_to_index(self, words, dictionary):
        """
        Replace each word in the dataset with its index in the dictionary """
        return [(dictionary[word] if word in dictionary else 0) for word in words]

    def generate_skip_grams(self, words_as_indices, context_window_size, printit=False):
        """ Form pairs according to the skip-gram model. """
        for index, center in enumerate(words_as_indices):
            for target in words_as_indices[max(0, index - context_window_size[0]):index]:
                if index % 10000 == 0 and printit:
                    print(index, target)
                yield (
                 center, target)

            for target in words_as_indices[index + 1:index + context_window_size[1] + 1]:
                if index % 10000 == 0 and printit:
                    print(index, target)
                yield (
                 center, target)

    def create_matrix(self, vocabulary_as_indices, context_words_as_indices, skip_grams):
        """ create the occurence matrix """
        occurence_matrix = np.zeros((len(vocabulary_as_indices),
         len(context_words_as_indices)))
        for center, target in skip_grams:
            occurence_matrix[(center, target)] += 1

        return occurence_matrix


def main():
    VOCAB_SIZE = 10000
    EMBED_SIZE = 50
    SKIP_WINDOW = (0, 1)
    matrix = CooccurenceMatrix(VOCAB_SIZE, EMBED_SIZE, SKIP_WINDOW)
    words = matrix.read_data('text8.zip')
    word_index, index_word = matrix.build_vocab(words, VOCAB_SIZE)
    words_as_indices = matrix.convert_words_to_index(words, word_index)
    skip_grams = matrix.generate_skip_grams(words_as_indices, SKIP_WINDOW, printit=False)
    occurence_matrix = matrix.create_matrix(list(index_word.keys()), list(index_word.keys()), skip_grams)
    row_sums = np.sum(occurence_matrix, axis=1)
    occurence_matrix_normalized = occurence_matrix / row_sums[:, np.newaxis]
    occ_ten = tf.Variable(occurence_matrix_normalized)
    s, u, v = tf.svd(occ_ten, full_matrices=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        occ_ten.initializer.run()
        singular_values_tf = s.eval()
        u_tf = u.eval()
        singular_values_np = np.asarray(singular_values_tf)
        u_tf = np.asarray(u_tf)
        np.savetxt('singular_values_np.csv', singular_values_np, delimiter=',')
        np.savetxt('u_tf.csv', u_tf, delimiter=',')


if __name__ == '__main__':
    main()
# okay decompiling context_counting_model.cpython-35.pyc
