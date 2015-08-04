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
 
def check_ref(label,ref_file):
    if (ref_file is None) or (ref_file == ''):
         return True; 
    
    fin = open(ref_file,'rb')
    for line in fin:
        line = line.strip()
        if (line == label):
            return True
    fin.close()
    return False
                        
def visualize(output_file_mix, output_file_pos,  model, ref_file, pos_dict, freq_dict, ref_pickle = ''):
    import pickle
    print ref_pickle
    if ref_pickle == '':
            size = 5000
            count = 0
            labels = [] 
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
            model_tsne = TSNE(n_components=2, random_state=0, perplexity = 20, n_iter = 1000)
            
            print (" Do TSNE ")
            Y = model_tsne.fit_transform(X_50)
            ref_pickle = "transform.pickle"
            print (" Saving to file :" + ref_pickle)
            need_save = (Y,labels,pos_dict)
            pickle.dump( need_save, open( ref_pickle, "wb" ) )
    
    # Load directly from pickle
    (Y,labels,pos_dict) = pickle.load( open(ref_pickle, "rb" ))
    # Show the scatter plot 
    do_visualization(Y, labels, pos_dict, ref_file, output_file_pos, output_file_mix)

def do_visualization(Y, labels, post_dict, ref_file, output_file_pos, output_file_mix):
    #import matplotlib
    #matplotlib.use('Agg')
    # First file 
    import matplotlib.pyplot as plt
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

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

    


    import matplotlib.pyplot as plt
    #frame1 = plt.gca()
    #frame1.axes.xaxis.set_ticklabels([])
    #frame1.axes.yaxis.set_ticklabels([])

    plt.scatter(Y[:,0], Y[:,1], s = 0.000000000000000000000001)
    
    # Add labels
    # Define pos and colour 
    pos_colour = {'ADJ':'black','ADP':'blue','PUNCT':'green','ADV':'Cyan','AUX':'Coral','SYM':'brown',
                            'INTJ':'darkviolet', 'CONJ':'purple', 'X':'lavender','NOUN':'red','DET':'yellow','PROPN':'goldenrod','NUM':'DarkMagenta','VERB':'chartreuse','PART':'burlywood',     
                            'PRON':'tan','SCONJ':'GreenYellow'}     
     
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        colour = 'black'
        text_vl = label
        pos = pos_dict[label]
        colour='white'
        for pos_vl in pos_colour.keys():
            if pos_vl == pos:
                colour = pos_colour[pos_vl]
        
        if check_ref(label,ref_file):
            plt.annotate(text_vl.decode('utf-8'), xy = (x,y), xytext = (0, 0), color = colour, textcoords = 'offset points', size = 2)    
    
    plt.savefig(output_file_pos, format='pdf')
    #plt.show()
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
    
    parser.add_argument('-f','--ref', help='Ref file', required=False)
    parser.add_argument('-ts','--stree', help='The source treebank', required=True)
    parser.add_argument('-tt','--ttree', help='The target treebank', required=True)
    parser.add_argument('-pf','--pfile', help='The pickle file')
    #parser.add_argument('-tt','--ttree', help='The target treebank', required=True)
    
    args = vars(parser.parse_args())
    input_file = args['input'];
    output_file = ""
    if args.has_key('output'):
        output_file = args['output'];
        
    pickle_file = ""
    if args.has_key('pfile'):
        pickle_file = args['pfile'];
        
    ref_file = ""; 
    if args.has_key('ref'):
        ref_file = args['ref']

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
     
    visualize(output_file+".mix.pdf", output_file + ".pos.pdf", model, ref_file, pos_dict, freq_dict, ref_pickle=pickle_file)
     

    
