from nltk.tokenize import word_tokenize
import csv

tok=[]
## Structure of the csv file
## Column 1 post title or writing post title
## Column 2 post author
## Column 3 comments body
## Column 4 comments author

## TODO: Create a list of .csv files and make a loop on it to process each file
##       Export data on .json format

with open('/home/harmodio/code/ganso/wp/data/csv/wp201601.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0 #to count how many lines are in a file
    for row in csv_reader:
            line_count += 1
            tok_row0 = [word_tokenize(row[0])]  ## tokenizes the title
            tok_row1 = [word_tokenize(row[1])]  ## tokenizes the author
            tok_row2 = [word_tokenize(row[2])]  ## tokenizes the body
            print(tok_row0)
            print(len(tok_row2[0]))  #to know the length of the token
            if len(tok_row2[0]) >= 30 and len(tok_row2[0]) <= 1000:
                ##This token is not going to be part of the final ones
                print('>30 <1000')


with open('/home/gerardo/Documents/Mios/Docsmios/AI/wp/wp201601.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
     print(tok_row0)
     #print(tok_row1)
     #print(tok_row2)
     #only 105 tokens
     #Excluir historias en donde una misma palabra se repita más de 10 veces (confirmer avec @jlr)_
    #*Lo que tenemos que hacer es remplazar las palabras que ocurren menos de 10 veces en el corpus por el token UNK (de _unknown_)* (edited)
    #6)
