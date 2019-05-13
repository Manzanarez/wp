import os
from io import open
import torch
from nltk.probability import FreqDist

class Dictionary(object):
#GAM Initializes selfwod2idx as a dictionary
#GAM and self.idx as a list

    def __init__(self):
#word2ixdx is a dictionary that contains a word and its index
        self.word2idx = {}
#idx2word contains the list of words
        self.idx2word = []
        self.vocabulary = []

    def morethan10(self,vocabulary_list):
        freq = FreqDist(vocabulary_list)
        #        print(freq.most_common((1000000)))
        indexv = 0
        for i in freq:
            if freq[i] < 10:
            ##         vocabulary_listl10 += vocabulary_list[indexv]
 ##               vocabulary_listl10.append(i)
##            else:
            ##        vocabulary_listm10 += vocabulary_list[indexv]
##                vocabulary_listm10.append(i)
##            indexv += 1
##            vocabulary_list[indexv] = "UNK"
                for w in range(len(self.idx2word)):
                    if self.idx2word[w] == i:
                        self.idx2word[w] = "UNK"
                        break
                self.word2idx[i]= -1
                print("Word less than 10: ", i, freq[i])
#                    for w in range(len(prompts_list)):
#                        for x in range(len(prompts_list[w])):
#                            if prompts_list[w][x] == i:
#                                prompts_list[w][x] = "UNK"
        return()

#GAM Adds a word to the dictionary and the list
    def add_word(self, word):
        self.vocabulary.append(word)
        if word not in self.word2idx:  #GAM ~ Only adds words that are unique in the vocabulary (not in word2idx)
            self.idx2word.append(word) #GAM ~ Appends a word to the list idx2word where the list number
                                       # is taken as the index (ex:idx2word[000000]={str}'Are')
            self.word2idx[word] = len(self.idx2word) - 1 ##GAM ~ Adds a word to the dictionary where every word has
                                                         #its index (ex: word2idx[Are] = {int} 0)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
#GAM Intializes self.dictionay as a Dictionary object
#GAM with selfwod2idx as a dictionary
#GAM and self.idx as a list
        self.dictionary = Dictionary()
##Original
##        self.trains = self.tokenize(os.path.join(path, 'train.txt'))
## GAM- Tokenizes lines available in the file and stores it in self.train (the corpus used to train)
#   GAM~ Makes a tensor (self.train) with the index of the words in the file used to train
##GAM May619 Activate
##        self.trainp = self.tokenize(os.path.join(path, 'prompts.tokens.alligned_train.09042019.txt')) ## Prompts to train
        self.trains = self.tokenize(os.path.join(path, 'stories.tokens.alligned_train.09042019.txt')) ## Stories to train
## Original
##        self.valids = self.tokenize(os.path.join(path, 'valid.txt'))
##        self.tests = self.tokenize(os.path.join(path, 'test.txt'))
#   GAM~ Makes a tensor (self.valid) with the index of the words in the file used to validate
##GAM May619 Activate
##        self.validp = self.tokenize(os.path.join(path, 'prompts.tokens.alligned_validate.09042019.txt')) #Prompts to validate
##GAM May619 Activate
        self.valids = self.tokenize(os.path.join(path, 'stories.tokens.alligned_validate.09042019.txt')) #Stories to validate
#   GAM~ Makes a tensor (self.valid) with the index of the words in the file used to test
##GAM May619 Activate
##        self.testp = self.tokenize(os.path.join(path, 'prompts.tokens.alligned_test.09042019.txt')) #Prompts to test
##GAM May619 Activate
        self.tests = self.tokenize(os.path.join(path, 'stories.tokens.alligned_test.09042019.txt')) #Stories to test

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
##                words = line.split() + ['<eos>'] ##GAM- Adds a '<eos>' at the end of the line but the file already has an EOD
                words = line.split()
                tokens += len(words) ## GAM ~ The quantity of words
                for word in words:
                    self.dictionary.add_word(word)  ## GAM ~ Adds a word that has from the corpus to the dictionary
                                                    ## GAM ~ idx2word ( list[] ) and word2idx ( dict{} )

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens) #GAM ~ Initilizes 'ids' as a tensor with data type 64-bit integer with the size of 'tokens'
            token = 0
            for line in f:  #GAM ~ 'f' is the file opened
##                words = line.split() + ['<eos>'] #GAM~ 'words' is a list that has the line of the file splited in words and adds <eos> to the end of the line
                words = line.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word] #GAM~ Saves all indixes that word2idx has for every 'word' in the line of 'f'
                                                                #GAM~ in "ids" tensor. Note: word2idx is a vocabulary with unique words
                                                                #GAM ~so "ids" has the index that correspond to that word in the line of 'f'
         ## GAM~ Remove           print ('self.dictionary.word2idx[word]: ', self.dictionary.word2idx[word])
         ## GAM~ Remove           print ('ids[token]: ', ids[token])
                    token += 1
## GAM ~ To find only words that are repeated more than 10 times        self.dictionary.morethan10(self.dictionary.vocabulary)
        print('ids:',ids)
        return ids