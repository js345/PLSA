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
        # random initialization of parameters
        self.pi_s = [np.random.dirichlet(np.ones(K)) for i in range(len(self.docs))]
        # K dictionaries of values sum to 1 in each
        self.theta_s = [{k: v for k, v in zip(self.word_dict,np.random.dirichlet(np.ones(len(self.word_dict))))} for i in range(K)]
        # generate background model
        total = sum(self.word_dict.values())
        self.word_dict = {k: v / total for k, v in self.word_dict.iteritems()}
        # log likelihood
        self.log_p = None

    def e_step(self):
        # np 2d array ndk dim |D| * K
        n_dk = np.zeros((len(self.docs), self.K))
        # nwk dim |V| * K dict of 1d array
        n_wk = {word: np.zeros(self.K) for word in self.word_dict}
        for k in range(self.K):
            for i in range(len(self.docs)):
                n_dk[i][k] = self.calculate_ndk(i, k)
            for word in self.word_dict:
                n_wk[word][k] = self.calculate_nwk(word, k)
        return n_dk, n_wk

    def m_step(self, n_dk, n_wk):
        # update pi
        for i in range(len(self.docs)):
            for k in range(self.K):
                self.pi_s[i][k] = n_dk[i][k] / sum(n_dk[i])
        # update theta
        for k in range(self.K):
            for word in self.theta_s[k]:
                self.theta_s[k][word] = n_wk[word][k] / sum([n_wk[word][k] for word in n_wk])

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
            log_diff = abs(log_p - self.log_p) / self.log_p
            log_graph.append(log_p)
            log_diff_graph.append(log_diff)
            if log_diff < diff:
                break
        return log_graph, log_diff_graph

    def compute_log(self):
        """
        Compute log likelihood
        :return:
        :rtype:
        """
        log_likelihood = 0
        for i in range(len(self.docs)):
            for j in range(len(self.docs[i])):
                inner_sum = 0
                for k in range(self.K):
                    inner_sum += self.pi_s[i][k] * self.theta_s[k][self.docs[i][j]]
                inner_sum = inner_sum * (1 - self.lamb) + self.lamb * self.word_dict[self.docs[i][j]]
                log_likelihood += inner_sum
        return log_likelihood

    def calculate_ndk(self, i, k):
        """
        Calculate ndk given document i and topic k
        :param i:
        :type i:
        :param k:
        :type k:
        :return:
        :rtype:
        """
        ndk = 0
        for word in self.word_count_list[i]:
            p_sum = 0
            for k_p in range(self.K):
                p_sum += self.pi_s[i][k_p] * self.theta_s[k_p][word]
            denominator = self.lamb * self.word_dict[word] + (1 - self.lamb) * p_sum
            nominator = (1 - self.lamb) * self.pi_s[i][k] * self.theta_s[k][word]
            ndk += self.word_count_list[i].get(word, 0) * nominator / denominator
        return ndk

    def calculate_nwk(self, word, k):
        """
        Calculate ndk given word and topic k
        :param word:
        :type word:
        :param k:
        :type k:
        :return:
        :rtype:
        """
        nwk = 0
        for i in range(len(self.word_count_list)):
            p_sum = 0
            for k_p in range(self.K):
                p_sum += self.pi_s[i][k_p] * self.theta_s[k_p][word]
            denominator = self.lamb * self.word_dict[word] + (1 - self.lamb) * p_sum
            nominator = (1 - self.lamb) * self.pi_s[i][k] * self.theta_s[k][word]
            nwk += self.word_count_list[i].get(word, 0) * nominator / denominator
        return nwk

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
