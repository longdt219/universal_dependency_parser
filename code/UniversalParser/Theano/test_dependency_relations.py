from math import sqrt
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy as np
import  operator 

def load_embedding_file(file_name):
    fin = open(file_name,"r")
    model = {}
    for line in fin:
        #line= line.strip()
        #print line
        tokens = line.split()
        #print (tokens[0])
        model[tokens[0]] = [float(x) for x in tokens[1:]]
        
    fin.close()
    return model 

def cosine_similarity(x,y):
    if len(x) != len(y):
        raise ValueError(" Dimension not matched ")
    sum = 0
    squarex= 0
    squarey = 0  
    for t in range(len(x)):
        sum += x[t] * y[t]
        squarex += x[t] * x[t]
        squarey += y[t] * y[t]
    return sum / (sqrt(squarex) * sqrt(squarey)) 
    
def check_sentence(model, sentence):
    tokens = sentence.split()
    for i in range(len(tokens) -1):
        for j in range(i+1,len(tokens)):
            if model.has_key(tokens[i]) and model.has_key(tokens[j]):
                print (tokens[i] + " -- " + tokens[j] + " : " + str(cosine_similarity(model[tokens[i]],model[tokens[j]])))
            else:
                raise ValueError(" Not contain in the model ")
        
def find_most_common(model, words):
    tokens = words.split()
    print (" FIND THE MOST COMMON WORDS")
    for token in tokens:
        token = 'en_' + token
        print (" WORD : " + token)
        if not model.has_key(token):
            raise ValueError(" Word not in the dictionary")
        
        sim = {}
        for token_ref in model.keys():
            if (token_ref != token):
                sim[token_ref] = cosine_similarity(model[token],model[token_ref])
        # Sort sim according to value 
        sorted_x = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(10):
            x,y = sorted_x[i]
            print x
            #print x

def find_most_common_shared(cross_model, mono_model, withdict_model, words, pickle_file = ""):
    
    labels = []
    if (pickle_file != ""):
        import pickle
        (Y,labels,pos_dict) = pickle.load( open(pickle_file, "rb" ))
    
    
    tokens = words.split()
    print (" FIND THE MOST COMMON WORDS")
    for token in tokens:
        print (" WORD : " + token)
        print ("----------------------------")
        print (" X-embedding without dict")
        print ("----------------------------")
        if not cross_model.has_key('en_' + token):
            raise ValueError(" Word not in the dictionary")
        if not mono_model.has_key(token):
            raise ValueError(" Word not in the dictionary")
        
        # For cross-lingual model
        print ("For English ") 
        sim = {} 
        for token_ref in cross_model.keys():
            if (token_ref.find('en_')!=-1) and (token_ref != 'en_'+ token):
                sim[token_ref] = cosine_similarity(cross_model['en_'+token],cross_model[token_ref])
        # Sort sim according to value 
        sorted_x = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(5):
            x,y = sorted_x[i]
            print x
            #print x
    
        print (" For French ")
        sim = {} 
        for token_ref in cross_model.keys():
            if (token_ref.find('fr_')!=-1) and (token_ref != 'en_'+ token):
                sim[token_ref] = cosine_similarity(cross_model['en_'+token],cross_model[token_ref])
        # Sort sim according to value 
        sorted_x = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(5):
            x,y = sorted_x[i]
            print x
            #print x

        if len(withdict_model) > 1:          
            print ("----------------------------")  
            print (" X-embedding with dict")
            print ("----------------------------")        
            # For cross-lingual model
            print ("For English ") 
            sim = {} 
            for token_ref in withdict_model.keys():
                if (token_ref.find('en_')!=-1) and (token_ref != 'en_'+ token):
                    if (len(labels) ==0):
                        sim[token_ref] = cosine_similarity(withdict_model['en_'+token],withdict_model[token_ref])
                    else:
                        if token_ref in labels:
                            sim[token_ref] = cosine_similarity(withdict_model['en_'+token],withdict_model[token_ref])
                            
            # Sort sim according to value 
            sorted_x = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
            for i in range(10):
                x,y = sorted_x[i]
                print x

            print (" For French ")
            sim = {} 
            for token_ref in withdict_model.keys():
                if (token_ref.find('fr_')!=-1) and (token_ref != 'en_'+ token):
                    if (len(labels) ==0):
                        sim[token_ref] = cosine_similarity(withdict_model['en_'+token],withdict_model[token_ref])
                    else:
                        if token_ref in labels:
                            sim[token_ref] = cosine_similarity(withdict_model['en_'+token],withdict_model[token_ref])

            # Sort sim according to value 
            sorted_x = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
            for i in range(10):
                x,y = sorted_x[i]
                print x
        
        print ("----------------------------")
        print (" Monolingual embedding")
        print ("----------------------------")
        # For monolingual model 
        sim = {}
        for token_ref in mono_model.keys():
            if (token_ref != token) and (cross_model.has_key('en_'+token_ref)):
                sim[token_ref] = cosine_similarity(mono_model[token],mono_model[token_ref])
        # Sort sim according to value 
        sorted_x = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(10):
            x,y = sorted_x[i]
            print x
            #print x

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters for the classifier ')
    parser.add_argument('-i','--input', help='Embedding input file ', required=True)
    parser.add_argument('-s','--sentence', help='Sentence to test ')
    parser.add_argument('-w','--words', help='Sets of words to inspect')
    parser.add_argument('-ii','--in2', help='Second set of embedding (original)')
    parser.add_argument('-iii','--in3', help='With dictionary kind of embedding')
    parser.add_argument('-pf','--pfile', help='Pickle file where there are list of word for representation')
    
    args = vars(parser.parse_args())
    input_file = args['input'];
    sentence = ""
    if args.has_key('sentence'):
        sentence = args['sentence'];
    words = "" 
    if args.has_key('words'):
        words = args['words']
    embedding_2 = ""
    
    if args.has_key('in2'):
        embedding_2 = args['in2']

    embedding_3 = ""        
    if args.has_key('in3'):
        embedding_3 = args['in3']
    
    pickle_file = ""
    if args.has_key('pfile'):
        pickle_file = args['pfile']
        
    # Read the embedding file 
    model = load_embedding_file(input_file)
    print (" Size of model " + str(len(model)))
    
    # Check the sentence
    
    if (sentence is not None):
        check_sentence(model, sentence)

    if (embedding_2 is not None):
        model2 = load_embedding_file(embedding_2)
        model3 = {}
        if embedding_3 is not None : 
            model3 = load_embedding_file(embedding_3)
            
        find_most_common_shared(model,model2,model3, words, pickle_file)
    else:
        if (words is not None): 
            find_most_common(model,words)
    
