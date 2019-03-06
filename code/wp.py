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
#file_list = glob.glob("/home/harmodio/code/ganso/wp/data/csv/*.csv")
file_list = glob.glob("/home/gerardo/code/wp/data/csv/*.csv")

for file_name in file_list:
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0 #to count how many lines are in a file
        print('PROCESSING FILE:', file_name)
    
        for row in csv_reader:
            line_count += 1
            tok_row0 = [word_tokenize(row[0])]  ## tokenizes the title
            tok_row1 = [word_tokenize(row[1])]  ## tokenizes the author
            tok_row2 = [word_tokenize(row[2])]  ## tokenizes the body

            print(len(tok_row2[0]))  #to know the length of the token
            if len(tok_row2[0]) >= 30 and len(tok_row2[0]) <= 1000:
                ##This token is going to be part of the final ones
                print(tok_row0[0]) #Printing the title
                print('title lenght:', len(tok_row0[0]))
                print(tok_row1[0]) #Printing the author
                print(tok_row2[0]) #Printing the body
                print('body lenght:', len(tok_row2[0]))
                vocabulary_list += tok_row0[0]
                vocabulary_list += tok_row2[0] ## in order to get the vocabulary
                print('Vocabulary list size', len(vocabulary_list))
unique_vocabulary_set = set (vocabulary_list)
print ('UNIQUE_VOCABULARY_SET:',sorted(unique_vocabulary_set))
print ('UNIQUE_VOCABULARY_SET_SIZE:',len(unique_vocabulary_set))
    
#with open('/home/gerardo/Documents/Mios/Docsmios/AI/wp/wp201601.csv') as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    for row in csv_reader:
#     print(tok_row0)
     #print(tok_row1)
     #print(tok_row2)
     #only 105 tokens
     #Excluir historias en donde una misma palabra se repita más de 10 veces (confirmer avec @jlr)_
    #*Lo que tenemos que hacer es remplazar las palabras que ocurren menos de 10 veces en el corpus por el token UNK (de _unknown_)* (edited)
    #6)
