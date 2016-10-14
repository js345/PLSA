'''
hw5 PLSA
Created on 10/13/16
@author: xiaofo
'''

from __future__ import division
import numpy as np


class PLSA:

    def __init__(self, K=20, lamb=0.5, filename='dblp-small.txt'):
        """
        :param K: num of topics
        :type K:
        :param lamb: probability lambda
        :type lamb:
        :param filename: data set filename
        :type filename:
        :return:
        :rtype:
        """
        self.K = K
        self.lamb = lamb
        self.filename = filename
        self.docs, self.word_count_list, self.word_dict = self.read_file(filename)
        # random initialization of parameters
        self.pi_s = [np.random.dirichlet(np.ones(K)) for i in range(len(self.docs))]
        self.theta_s = [np.random.dirichlet(np.ones(len(self.word_dict))) for i in range(K)]
        # generate background model
        total = sum(self.word_dict.values())
        self.word_dict = {k: v / total for k, v in self.word_dict.iteritems()}

    def e_step(self):
        pass

    def m_step(self):
        pass

    @staticmethod
    def read_file(filename):
        """
        Parse input file
        :param filename:
        :type filename:
        :return:
        :rtype:
        """
        docs = list()
        word_count_list = list()
        word_dict = dict()
        f = open(filename, 'r')
        for line in f.readlines():
            doc = line.strip().split(' ')
            word_count = dict()
            for word in doc:
                word_count[word] = word_count.get(word, 0) + 1
                word_dict[word] = word_dict.get(word, 0) + 1
            docs.append(doc)
            word_count_list.append(word_count)
        return docs, word_count_list, word_dict

if __name__ == '__main__':
    plsa = PLSA()