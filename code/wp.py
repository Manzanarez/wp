from typing import TextIO

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, RegexpTokenizer
from langdetect import detect
import string
import csv
import glob
import re
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('info.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)



## Structure of the csv file
## Column 1 post title or writing post title
## Column 2 post author
## Column 3 comments body
## Column 4 comments author

## TODO: Create a list of .csv files and make a loop on it to process each file
##       Export data on .json format

#with open('/home/harmodio/code/ganso/wp/data/csv/wp201601.csv') as csv_file:
#with open('/home/gerardo/code/wp/data/csv/wp201601.csv') as csv_file:
    
#TODO: For on all corpus files

#print(glob.glob("/home/gerardo/code/wp/data/csv/*.csv")
freq = 0 ## frequency distribution
vocabulary_list = []
vocabulary_listm10 = []
vocabulary_listl10 = []
prompts_list = []
stories_list = []
prompts_indexl = []
corpus_size = 0
notvalidflairt = 0
offtopict = 0
#lst = list()
prof1t = 0
prof2t = 0
prof3t = 0
prof0t = 0
bott = 0
notin301000t = 0
langnotent = 0
erlangnotent = 0
notinflairlt = 0
line_countt = 0
indexv = 0
wordnum = 0
#laptop directory
##dir = "/home/gerardo/code/wp/"
#server directory
dir = "/users/aleman/wp/"


##  file_list = glob.glob("/home/harmodio/code/ganso/wp/data/csv/*.csv")
#laptop directory
##file_list = glob.glob("/home/gerardo/code/wp/data/csv/*.csv")

#directory of .csv files on the server
##file_list = glob.glob("/users/aleman/wp/data/csv/wp2016*.csv")
#file to test on the server
##file_list = glob.glob("/users/aleman/wp/data/csv/*wp2017w11.csv")

#files to read from writing prompts per year/week
file_list = glob.glob(dir+"data/csv/*.csv")

##  file_list = glob.glob("/home/gerardo/code/wp/data/csv/wp2016w33.csv")
##  file_list = glob.glob("/home/gerardo/code/wp/data/test/wp2017w11.csv")
##file_list = glob.glob("/home/gerardo/code/wp/data/train/*.csv") ## Files to train
##  file_list = glob.glob("/home/gerardo/code/wp/data/test/wp2016w41.csv")

##file in the server
##vocab = open("/users/aleman/wp/data/train/vocab.txt","w+") ##File that stores the unique vocabulary

##file in my computer
vocab = open(dir+"data/train/vocab.txt","w+") ##File that stores the unique vocabulary

##promptfile = open("pf.txt","w+") ##File to write all the prompts
##storyfile = open("sf.txt","w+") ##File to write all the prompts stories
#laptop directory
promptfilel = open(dir+"data/train/prompts.tokens.alligned_train.18092019.txt","w+") ##File to write all the prompts
storyfilel = open(dir+"data/train/stories.tokens.alligned_train.18092019.txt","w+") ##File to write all the prompts stories

##server directories
##promptfilel = open("/users/aleman/wp/data/train/prompts.tokens.alligned_train.092019.txt","w+") ##File to write all the prompts
##storyfilel = open("/users/aleman/wp/data/train/stories.tokens.alligned_train.092019.txt","w+") ##File to write all the prompts stories

#laptop directory
analysis_result = open(dir+"data/train/analysis_resultTest.txt","w+") ##File to write the analysis of all the files read
word_repeatstats = open(dir+"data/train/word_repeatstatsTest.txt", "w+") ## File to write the analysis of all the words (reapeated)

##server directories
##analysis_result = open("/users/aleman/wp/data/train/analysis_resultTest.txt","w+") ##File to write the analysis of all the files read
##word_repeatstats = open("/users/aleman/wp/data/train/word_repeatstatsTest.txt", "w+") ## File to write the analysis of all the words (reapeated)

##Do not consider promts that are labeled as:
##[MP] = Media Prompt: Audio or Video
##[IP] = Image Prompt: A striking image or album
##[OT] = Off Topic: Not a prompt, but writing related
##[MOD] = Moderator Post: Important announcements or events
## Or do not have labels enclosed in '[]'
## **Off Topice in the body text

valid_prompts = ['WP','SP','EU','CW','TT','RF','PM','PI','CC']
##with open("base-list-of-bad-words_CSV-file_2018_07_30.csv") as prof_file:
 ##   prof_filer = csv.reader(prof_file, delimiter=',')
#with open(base-list-of-bad-words_CSV-file_2018_07_30.csv) as f:
#    f = f.readlines()
tok_rowr = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
#prof="Fuck"
for file_name in file_list:
    with open(file_name) as csv_file:

        logger.info('Beginning of file %s',file_name)
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0 #to count how many lines are in a file
        notvalidflair = 0
        offtopic = 0
        prof1 = 0
        prof2 = 0
        prof3 = 0
        prof0 = 0
        bot = 0
        notin301000 = 0
        langnoten = 0
        erlangnoten = 0
        notinflairl = 0
        print('PROCESSING FILE:', file_name)
        word_dist = FreqDist()
        prompt_index = 0
        storie_index = 0

        for row in csv_reader:
            tokenx = word_tokenize(row[2])
#            ll = [x for x in l if not re.fullmatch('[' + string.punctuation + ']+', x)]
#           tok_row2t = [x for x in tokenx if not re.fullmatch('[' + string.punctuation + ']+',x)]
#           tok_row2t = [x for x in tok_row2t if re.fullmatch('[' + string.ascii_letters + ']+',x)]
##            tok_row0 = ["IND"] #Index in tok_row0 list first element
            tok_row0 = ["IND"] + [tok_rowr.tokenize(row[0])]  ## tokenizes the title
#            tok_row1 = [tok_rowr.tokenize(row[1])]  ## tokenizes the author
##            tok_row2 = ["IND"]  # Index in tok_row2 list first element
            tok_row2 = ["IND"] + [tok_rowr.tokenize(row[2])]  ## tokenizes the body
#            tok_row0 = [word_tokenize(row[0])]  ## tokenizes the title
#            tok_row1 = [word_tokenize(row[1])]  ## tokenizes the author
#            tok_row2 = [word_tokenize(row[2])]  ## tokenizes the body
            line_count += 1
            line_countt = line_countt + 1

##            print(len(tok_row2[0]))  #to know the length of the token
##          Valid flair list
            if len(tok_row2[1]) >= 30 and len(tok_row2[1]) <= 1000:
                try:
                    if tok_row0[1][0] == 'WP' or tok_row0[1][0] == 'SP' \
                    or tok_row0[1][0] == 'EU' or tok_row0[1][0] == 'CW' \
                    or tok_row0[1][0] == 'TT' or tok_row0[1][0] == 'RF' \
                    or tok_row0[1][0] == 'TT' or tok_row0[1][0] == 'RF' \
                    or tok_row0[1][0] == 'PM' or tok_row0[1][0] == 'PI' \
                    or tok_row0[1][0] == 'CC' or tok_row0[1][0] == 'MP' \
                    or tok_row0[1][0] == 'IP':
                    ##               and tok_row0[0][1] in valid_prompts:

                        if tok_row2[1][0] == '**Off-Topic' or tok_row2[1][0] == 'Off topic':  # Do not consider body that contains ** Off Topic
                            offtopic += 1
                            offtopict = offtopict + 1
##24sep19                            print('**Off-Topic')
                        else:
                            #Do not consider if the submission has been remover
                            if 'www.reddit.com/r/WritingPrompts/' in row[2]:
##24sep19                                 print("Submission removed by WritingPrompts")
                                bot += 1
                                bott = bott + 1
                            else:
                                #Do not consider if The post has been removed
                                if 'has been removed' in row[2]:
##24sep19                                     print("Post has been removed by WritingPrompts ")
                                    bot += 1
                                    bott = bott + 1
                                else:
                                    prof0 = 0 #to know if the text has profanity
                                    try:
                                        language = detect(row[2]) #To detect the language of the prompt
                                        if language == 'en': #only consider prompts in english
##                                            line_count += 1
                                        #list of words with profanity
                                        ## server directory
##                                            with open("/users/aleman/wp/data/validate/bad_words/base-list-of-bad-words_val.csv",encoding = "ISO-8859-1") as prof_file:
                                        ##laptop directory
                                            with open(dir+"data/validate/bad_words/base-list-of-bad-words_val.csv",encoding = "ISO-8859-1") as prof_file:

##            with open("base-list-of-bad-words_TXT-file_2018_07_30.txt",encoding = "ISO-8859-1") as prof_file:
                                                prof_filer = csv.reader(prof_file, delimiter=',')
                                                for prof_word in prof_filer:
                                                    #search profanity in lower case. Note: using re.search() to find the whole word
                                                    if re.search(r'\b' + prof_word[0] + r'\b', row[0]) or re.search(r'\b' + prof_word[0] + r'\b', row[2]):
##24sep19                                                         print("Profanity1", prof_word[0])
                                                        prof1 += 1
                                                        prof1t = prof1t + 1
                                                        prof0 = 1
                                                        break
                                                    else:
                                                    #Convert to upper case the whole word
                                                        prof_word_upper = [i.upper() for i in prof_word]
                                                        # search profanity word in upper case . Note: using re.search() to find the whole word
                                                        if re.search(r'\b' + prof_word_upper[0] + r'\b', row[0]) or re.search(r'\b' + prof_word_upper[0] + r'\b', row[2]):
##24sep19                                                             print("Profanity2", prof_word_upper[0])
                                                            prof2 += 1
                                                            prof2t = prof2t + 1
                                                            prof0 = 1
                                                            break
                                                        else:
                                                            #Capitalize only the first letter of the word
                                                            prof_word_upper = [i.capitalize() for i in prof_word]
                                                            #search profanity word in upper case . Note: using re.search() to find the whole word
                                                            if re.search(r'\b' + prof_word_upper[0] + r'\b', row[0]) or re.search(r'\b' + prof_word_upper[0] + r'\b', row[2]):
##24sep19                                                                 print("Profanity3", prof_word_upper[0])
                                                                prof3 += 1
                                                                prof3t = prof3t + 1
                                                                prof0 = 1
                                                                break
                                                if prof0 == 0:
                                                    corpus_size+=1
                                                    ##"EOD" token added to end of the body
##                                                    tok_row0[0].append('\n')
                                                    tok_row2[1].append('EOD')
##                                                    tok_row2[0].append("\n")
##                                                    tok_row2[0].append('\n')
                ##This token is going to be part of the final ones
                ## Printing the prompt title
##24sep19                                                     print(tok_row0[1])

##                    print(tok_row1[0]) #Printing the author  ## To know the author
##24sep19                                                     print(tok_row2[1]) #Printing the body
##                print('body lenght:', len(tok_row2[0]))
##                                                    promptfile.write("{}\n".format(tok_row0[0]))
##                                                    storyfile.write("{}\n".format(tok_row2[0]))

                                                    vocabulary_list += tok_row0[1]
                                                    vocabulary_list += tok_row2[1] ## in order to get the vocabulary
                                                    storie_index += 1
                                                    if prompt_index > 0:
                                                        for ind in range(len(prompts_indexl)):
                                                            if prompts_indexl[ind] == tok_row0[1]:
                                                                break
                                                            else:
                                                                if ind+1 == len(prompts_indexl):
                                                                    prompt_index += 1
                                                                    prompts_indexl += [tok_row0[1]]
##                                                    prompts_list += [str(prompt_index)] + [tok_row0[0]]
                                                    tok_row0[0] = str(prompt_index)
                                                    tok_row2[0] = str(prompt_index)
                                                    prompts_list += [tok_row0]
                                                    stories_list += [tok_row2]
                                                    if prompt_index == 0:
                                                        prompt_index += 1
                                                        prompts_indexl +=  [tok_row0[1]]
##                print('Vocabulary list size', len(vocabulary_list))
                                        else:
                                            langnoten += 1
                                            langnotent = langnotent + 1
##24sep19                                             print('Language not english')
                                    except:
                                        erlangnoten += 1
                                        erlangnotent = erlangnotent + 1
##24sep19                                         print('Error Language not english')
                    else:
                        notinflairl +=1
                        notinflairlt = notinflairlt + 1
##24sep19                         print('Not in flair list')
                except IndexError:
                    notvalidflair += 1
                    notvalidflairt = notvalidflairt + 1
            else:
                notin301000 += 1
                notin301000t = notin301000t + 1
 ## server directory
##        with open("/users/aleman/wp/data/train/analysis_resultTest.txt", "a+") as analysis_result:
        with open(dir+"data/train/analysis_resultTest.txt", "a+") as analysis_result:
 ##         analysis_result.write('UNIQUE_VOCABULARY_SET_SIZE: %d \n' % len(unique_vocabulary_set))
            analysis_result.write('PROCESSING FILE: %s \n' % file_name)
            analysis_result.write('CORPUS_SIZE: %d \n' % corpus_size)
            analysis_result.write('LINE COUNT: %d \n' % line_count)
            analysis_result.write('NOT IN FLAIR LIST: %d \n' % notinflairl)
            analysis_result.write('BOT: %d \n' % bot)
            analysis_result.write('NOT VALID FLAIR: %d \n' % notvalidflair)
            analysis_result.write('OFF-TOPIC: %d \n' % offtopic)
            analysis_result.write('PROFANITY lower case: %d \n' % prof1)
            analysis_result.write('PROFANITY CAPITALS: %d \n' % prof2)
            analysis_result.write('Profanity First Letter Capital: %d \n' % prof3)
            logger.info('End of file %s', file_name)

##24sep19 print(vocabulary_list)
freq = FreqDist(vocabulary_list)
##24sep19 print(freq.most_common((1000000)))
indexv = 0
logger.info('Beginning of word repeat writing %s',word_repeatstats)
for i in freq:
    if freq[i] == 1:
##24sep19         print('The word ', i, ' repeats ', freq[i], ' time. \n')
        word_repeatstats.write('The word %s repeats %d time.\n' % (i, freq[i]))
    else:
##24sep19         print ('The word ', i , ' repeats ' , freq[i] , ' times. \n')
        word_repeatstats.write ('The word %s repeats %d times.\n' % (i,freq[i]))

    if freq[i] < 10:
##         vocabulary_listl10 += vocabulary_list[indexv]
        vocabulary_listl10.append(i)
    else:
##        vocabulary_listm10 += vocabulary_list[indexv]
        vocabulary_listm10.append(i)
    indexv +=1
#        vocabulary_list[indexv] = "UNK"
#        for w in range(len(stories_list)):
#            for x in range(len(stories_list[w])):
#                if stories_list[w][x] == i:
#                    stories_list[w][x] = "UNK"
#        for w in range(len(prompts_list)):
#           for x in range(len(prompts_list[w])):
#                if prompts_list[w][x] == i:
#                    prompts_list[w][x] = "UNK"
logger.info('End of word repeat writing %s',word_repeatstats)

##        prompts_list[indexv] = "UNK"
##        stories_list[indexv] = "UNK" ## Find the word in all the lists

##print(prompts_list) > promptfilel
logger.info('Beginning of writing prompts tokenized %s',promptfilel)
for i in range(len(prompts_list)):
    promptfilel.write("{}\n".format(prompts_list[i]))
logger.info('End of writing prompts tokenized %s',promptfilel)

logger.info('Beginning of writing stories tokenized %s',storyfilel )
for i in range(len(stories_list)):
    storyfilel.write("{}\n".format(stories_list[i]))
logger.info('End of writing stories tokenized %s',storyfilel)
#promptfilel.write(" ".join(prompts_list))
#storyfilel.write(" ".join(stories_list))

unique_vocabulary_set = set(vocabulary_listm10)
#vocab_tot = open("vocab_tot.txt","w")
## server directory
##vocab_less10 = open("/users/aleman/wp/data/train/vocab_less10Test.txt","w")
#laptop directory
vocab_less10 = open(dir+"data/train/vocab_less10Test.txt","w")
#vocab_tot.write(vocab)
##server directory
##vocab = open("/users/aleman/wp/data/train/vocabTest.txt","w")
##laptop directory
vocab = open(dir+"data/train/vocabTest.txt","w")
for v in sorted(unique_vocabulary_set):
    vocab.write("{}\n".format(v))

for v in range(len(vocabulary_listl10)):
    vocab_less10.write("{}\n".format(vocabulary_listl10[v]))

#for w in stories_list:
#    if w == "EOD":
#        storyfilel.write("{}\n"(stories_list[wordnum]))
#    else:
#        storyfilel.write("{}".format(stories_list[wordnum]))
#    wordnum += 1

#wordnum = 0
#for w in prompts_list:
#    if w == "EOD":
#        promptfilel.write("{}\n".format(prompts_list[wordnum]))
#    else:
#        promptfilel.write("{}".format(prompts_list[wordnum]))
#    wordnum += 1

##print ('UNIQUE_VOCABULARY_SET:',sorted(unique_vocabulary_set)) #Is already in file vocab.txt
print ('UNIQUE_VOCABULARY_SET_SIZE:',len(unique_vocabulary_set))
print ('CORPUS_SIZE:', corpus_size)
print ('Line count: ', line_countt)
print ('Language not english: ', langnotent)
print ('Language not english error: ', erlangnotent)
print ('Not in 30 and 1000: ', notin301000t)
print ('Bot: ',bott)
print ('Not in flair list: ', notinflairlt)
print ('Not valid flair: ', notvalidflairt)
print ('Off-topic: ', offtopict)
print ('Profanity lower case: ', prof1t)
print ('Profanity CAPITALS: ', prof2t)
print ('Profanity First Letter Capital: ', prof3t)

with open(dir+"data/train/analysis_resultTest.txt","a+") as analysis_result:
    analysis_result.write('\n********FINAL RESULT********* \n')
    analysis_result.write('UNIQUE_VOCABULARY_SET_SIZE: %d \n' % len(unique_vocabulary_set))
##   analysis_result.write('PROCESSING FILE: %s \n' % file_name)
    analysis_result.write('CORPUS_SIZE: %d \n' % corpus_size)
    analysis_result.write('LINE COUNT: %d \n' % line_countt)
    analysis_result.write('BOT: %d \n' % bott)
    analysis_result.write('NOT FLAIR LIST: %d \n' % notinflairlt)
    analysis_result.write('NOT VALID FLAIR: %d \n' % notvalidflairt)
    analysis_result.write('OFF-TOPIC: %d \n' % offtopict)
    analysis_result.write('PROFANITY lower case: %d \n' % prof1t)
    analysis_result.write('PROFANITY CAPITALS: %d \n' % prof2t)
    analysis_result.write('Profanity First Letter Capital: %d \n' % prof3t)
    logger.info('End of all files')
