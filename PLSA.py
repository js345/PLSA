'''
hw5 PLSA
Created on 10/13/16
@author: xiaofo
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


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
        print len(self.docs), len(self.word_count_list), len(self.word_dict)
        # random initialization of parameters
        self.pi_s = [np.random.dirichlet(np.ones(K)) for i in range(len(self.docs))]
        # K dictionaries of values sum to 1 in each
        self.theta_s = [{k: v for k, v in zip(self.word_dict, np.random.dirichlet(np.ones(len(self.word_dict))))} for i in range(K)]
        # generate background model
        total = sum(self.word_dict.values())
        self.word_dict = {k: v / total for k, v in self.word_dict.iteritems()}
        # log likelihood
        self.log_p = None
        # memoization
        # inner_sum structure {word : np.zeros(len(self.docs)) for word in self.word_dict}
        self.inner_sum = dict()
        # outer_sum structure {word : np.zeros(len(self.docs),self.K) for word in self.word_dict}
        self.outer_sum = dict()

    def e_step(self):
        """
        E step compute two counts
        :return: n_dk, n_wk
        :rtype:
        """
        # np 2d array ndk dim |D| * K
        n_dk = np.ones((len(self.docs), self.K))
        # nwk dim |V| * K dict of 1d array
        n_wk = {word: np.ones(self.K) for word in self.word_dict}
        for k in range(self.K):
            for i in range(len(self.docs)):
                n_dk[i][k] = sum(self.outer_sum[w][i][k] for w in self.outer_sum)
            for word in self.word_dict:
                n_wk[word][k] = sum(self.outer_sum[word][index][k] for index in range(len(self.docs)))
        return n_dk, n_wk

    def m_step(self, n_dk, n_wk):
        """
        Using counts to update parameters
        :param n_dk:
        :type n_dk:
        :param n_wk:
        :type n_wk:
        :return:
        :rtype:
        """
        for i in range(len(self.docs)):
            n_dk_sum = sum(n_dk[i])
            # update pi
            for k in range(self.K):
                self.pi_s[i][k] = n_dk[i][k] / n_dk_sum

        for k in range(self.K):
            n_wk_sum = sum(n_wk[w][k] for w in n_wk)
            # update theta
            for word in self.theta_s[k]:
                self.theta_s[k][word] = n_wk[word][k] / n_wk_sum

    def pre_process(self):
        """
        pre compute inner and outer sums to improve performance
        :return:
        :rtype:
        """
        # inner_sum structure {word : np.zeros(len(self.docs)) for word in self.word_dict}
        self.inner_sum = dict()
        # outer_sum structure {word : np.zeros(len(self.docs),self.K) for word in self.word_dict}
        self.outer_sum = dict()
        # memoization on inner_sum
        for word in self.word_dict:
            self.inner_sum[word] = np.zeros(len(self.docs))
            # memoization on outer_sum based on inner_sum
            self.outer_sum[word] = np.zeros((len(self.docs), self.K))
            for i in range(len(self.docs)):
                p_sum = 0
                for k_p in range(self.K):
                    p_sum += self.pi_s[i][k_p] * self.theta_s[k_p][word]
                self.inner_sum[word][i] = p_sum
                # for outer sum
                count = self.word_count_list[i].get(word, 0)
                # no word count means 0 for all k
                if count == 0:
                    continue
                for k in range(self.K):
                    denominator = self.lamb * self.word_dict[word] + (1 - self.lamb) * self.inner_sum[word][i]
                    nominator = (1 - self.lamb) * self.pi_s[i][k] * self.theta_s[k][word]
                    self.outer_sum[word][i][k] = count * nominator / denominator
        print "Preprocess done"

    def run(self, iteration=100, diff=0.0001):
        self.log_p = self.compute_log()
        log_graph, log_diff_graph = list(), list()
        for i in range(iteration):
            print "Iteration " + str(i)
            n_dk, n_wk = self.e_step()
            print "E step done"
            self.m_step(n_dk, n_wk)
            print "M step done"
            log_p = self.compute_log()
            log_diff = abs(self.log_p - log_p) / self.log_p
            self.log_p = log_p
            log_graph.append(log_p)
            log_diff_graph.append(log_diff)
            if log_diff < diff:
                pass
        return log_graph, log_diff_graph

    def compute_log(self):
        """
        Compute log likelihood
        :return:
        :rtype:
        """
        self.pre_process()
        log_likelihood = 0
        for i in range(len(self.docs)):
            for j in range(len(self.docs[i])):
                inner_sum = self.inner_sum[self.docs[i][j]][i]
                total = inner_sum * (1 - self.lamb) + self.lamb * self.word_dict[self.docs[i][j]]
                log_likelihood += total
        return log_likelihood

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
    plsa = PLSA(20, 0.9)
    log_graph, diff_graph = plsa.run()
    for topic in plsa.theta_s:
        print sorted(topic.items(), key=lambda x: x[1], reverse=True)[:10]
    plt.plot(log_graph)
    plt.title("Log likelihood plot")
    plt.show()
    plt.plot(diff_graph)
    plt.title("Log differnece plot")
    plt.show()
