__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy as np
import gensim 

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

def visualize(output_file_mix, output_file_pos,  model, size, pos_dict, freq_dict):
    labels = [] 
    count = 0
    
    if size >  len(model):
        size = len(model)
             
    max_count = size
    
    X = np.zeros(shape=(max_count,len(model[list(model.keys())[0]])))
    
    for term in model.keys():
        freq_dict.setdefault(term, 0) 
        if freq_dict[term] > 4:
            X[count] = model[term]
            labels.append(term)    
            count+= 1 
            if count >= max_count: break     
    
    # It is recommended to use PCA first to reduce to ~50 dimensions  
    from sklearn.decomposition import PCA
    print (" Do PCA ")
    pca = PCA(n_components=50)
    X_50 = pca.fit_transform(X)
    
    # Using TSNE to further reduce to 2 dimensions
    from sklearn.manifold import TSNE  
    model_tsne = TSNE(n_components=2, random_state=0, perplexity = 20)
    
    print (" Do TSNE ")
    Y = model_tsne.fit_transform(X_50)
    
    # Show the scatter plot 
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    plt.scatter(Y[:,0], Y[:,1], s = 0.000000000000000000000001)
    
    # Add labels 
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        colour = 'black'
        text_vl = label[3:] 
        if label.find('fr_'):
             colour = 'red'
        if label.find('en_'):
             colour = 'green'
             
        #print (text_vl)
        plt.annotate(text_vl.decode('utf-8'), xy = (x,y), xytext = (0, 0), color = colour, textcoords = 'offset points', size = 2)    
    
    plt.savefig(output_file_mix, format='pdf')
    plt.close()
    # Save to different file 

    import matplotlib.pyplot as plt
    plt.scatter(Y[:,0], Y[:,1], s = 0.000000000000000000000001)
    
    # Add labels
    # Define pos and colour 
    pos_colour = {'ADJ':'black','ADP':'blue','PUNCT':'green','ADV':'Cyan','AUX':'Coral','SYM':'brown',
                            'INTJ':'darkviolet', 'CONJ':'purple', 'X':'lavender','NOUN':'red','DET':'yellow','PROPN':'goldenrod','NUM':'DarkMagenta','VERB':'chartreuse','PART':'burlywood',     
                            'PRON':'tan','SCONJ':'GreenYellow'}     
     
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        colour = 'black'
        text_vl = label[3:]
        pos = pos_dict[label]
        colour='white'
        for pos_vl in pos_colour.keys():
            if pos_vl == pos:
                colour = pos_colour[pos_vl]
        
        plt.annotate(pos, xy = (x,y), xytext = (0, 0), color = colour, textcoords = 'offset points', size = 2)    
    
    plt.savefig(output_file_pos, format='pdf')
    plt.close()


def read_treebank(in_file, prefix, pos_dict = {}, freq_dict = {}):
    fin = open(in_file,'rb')
    post_dict_temp = {}
    for line in fin:
        line = line.strip()
        if line is None : continue 
        tokens= line.split()
        if len(tokens) < 5 : continue 
        word = tokens[1]
        pos = tokens[3]
        word = prefix + word
        
        freq_dict.setdefault(word,0)
        freq_dict[word] += 1
        
        post_dict_temp.setdefault(word,{})
        post_dict_temp[word].setdefault(pos,0) 
        post_dict_temp[word][pos] +=1
            
    fin.close()
    # Only get the highest one 
    for word in post_dict_temp.keys():
        max = 0 
        pos = '' 
        for key in post_dict_temp[word].keys():
            if post_dict_temp[word][key] > max:
                max = post_dict_temp[word][key]
                pos = key
        pos_dict[word] = pos
    
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters for the classifier ')
    parser.add_argument('-i','--input', help='Embedding input file ', required=True)
    parser.add_argument('-o','--output', help='Output pdf file')
    
    parser.add_argument('-s','--size', help='Limit size of input for display', required=True)
    parser.add_argument('-ts','--stree', help='The source treebank', required=True)
    parser.add_argument('-tt','--ttree', help='The target treebank', required=True)
    
    args = vars(parser.parse_args())
    input_file = args['input'];
    output_file = ""
    if args.has_key('output'):
        output_file = args['output'];
        
    size = 1000 
    if args.has_key('size'):
        size = int(args['size'])
    source_tree = args['stree']
    target_tree = args['ttree']
    
    # Read the embedding file 
    model = load_embedding_file(input_file)
    print (" Size of model " + str(len(model)))
    
    # Read treebank 
    pos_dict = {} 
    freq_dict = {}
    read_treebank(source_tree, "en_", pos_dict, freq_dict)
    read_treebank(target_tree, "fr_", pos_dict, freq_dict)
    
    # START FROM HERE, ONLY VISUALIZE WORDS THAT HAVING POS .... and FREQUENCY .....
    
    # Visualize it
     
    visualize(output_file+".mix.pdf", output_file + ".pos.pdf", model, size, pos_dict, freq_dict)
     

    
