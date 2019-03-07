import nltk
a = "Guru99 is the site where you can find the best tutorials for Software Testing     Tutorial, SAP Course for Beginners. Java Tutorial for Beginners and much more. Please     visit the site guru99.com and much more."
words = nltk.tokenize.word_tokenize(a)
fd = nltk.FreqDist(words)
#fd.plot()

print(words,fd.values())
fd.pprint()
#for n in len(words):
#    if fd[n] > 3:
#        print(fd[words[n]])