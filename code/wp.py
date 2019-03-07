from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import csv
import glob


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
vocabulary_list = []
corpus_size = 0
notvalidflair = 0
offtopic = 0
#file_list = glob.glob("/home/harmodio/code/ganso/wp/data/csv/*.csv")
#file_list = glob.glob("/home/gerardo/code/wp/data/csv/*.csv")
file_list = glob.glob("/home/gerardo/code/wp/data/csv/wp2016w33.csv")

vocab = open("vocab.txt","w+") ##File that stores the unique vocabulary

##Do not consider promts that are labeled as:
##[MP] = Media Prompt: Audio or Video
##[IP] = Image Prompt: A striking image or album
##[OT] = Off Topic: Not a prompt, but writing related
##[MOD] = Moderator Post: Important announcements or events
## Or do not have labels enclosed in '[]'
## **Off Topice in the body text

valid_prompts = ['WP','SP','EU','CW','TT','RF','PM','PI','CC']

#with open(base-list-of-bad-words_CSV-file_2018_07_30.csv) as f:
#    f = f.readlines()

for file_name in file_list:
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0 #to count how many lines are in a file
        print('PROCESSING FILE:', file_name)
        word_dist = FreqDist()
    
        for row in csv_reader:
            line_count += 1
            tok_row0 = [word_tokenize(row[0])]  ## tokenizes the title
            tok_row1 = [word_tokenize(row[1])]  ## tokenizes the author
            tok_row2 = [word_tokenize(row[2])]  ## tokenizes the body

##            print(len(tok_row2[0]))  #to know the length of the token

            if len(tok_row2[0]) >= 30 and len(tok_row2[0]) <= 1000:
                try:
                    tok_row0[0][1] == 'WP' or tok_row0[0][1] == 'SP'\
                    or tok_row0[0][1] == 'EU' or tok_row0[0][1] == 'CW'\
                    or tok_row0[0][1] == 'TT' or tok_row0[0][1] == 'RF' \
                    or tok_row0[0][1] == 'PM' or tok_row0[0][1] == 'PI' \
                    or tok_row0[0][1] == 'CC' or tok_row0[0][1] == 'MP' \
                    or tok_row0[0][1] == 'IP'
                                ##               and tok_row0[0][1] in valid_prompts:

                    if tok_row2[0][0] == '**Off-Topic':  #Do not consider body that contains ** Off Topic
                        offtopic += 1
                    else:
                        corpus_size+=1
                ##This token is going to be part of the final ones

                        print(tok_row0[0]) #Printing the title
##                print('title lenght:', len(tok_row0[0]))
##                    print(tok_row1[0]) #Printing the author  ## To know the author
                        print(tok_row2[0]) #Printing the body
##                print('body lenght:', len(tok_row2[0]))
                        vocabulary_list += tok_row0[0]
                        vocabulary_list += tok_row2[0] ## in order to get the vocabulary
##                print('Vocabulary list size', len(vocabulary_list))

                except IndexError:
                    notvalidflair += 1
unique_vocabulary_set = set (vocabulary_list)
vocab = open("vocab.txt","w")
for v in sorted(unique_vocabulary_set):
    vocab.write("{}\n".format(v))


print ('UNIQUE_VOCABULARY_SET:',sorted(unique_vocabulary_set))
print ('UNIQUE_VOCABULARY_SET_SIZE:',len(unique_vocabulary_set))
print ('CORPUS_SIZE:', corpus_size)
