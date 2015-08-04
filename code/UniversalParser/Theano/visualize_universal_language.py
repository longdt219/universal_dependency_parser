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

def visualize(output_file, model, size):
    labels = [] 
    count = 0
    
    if size >  len(model):
        size = len(model)
             
    max_count = size
    
    X = np.zeros(shape=(max_count,len(model[list(model.keys())[0]])))
    
    for term in model.keys():
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
    # START FROM HERE 
    
    
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    plt.scatter(Y[:,0], Y[:,1], s = 0.00000000000000000000000001)
    
    # Add labels 
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]): 
        plt.annotate(label.decode("utf-8"), xy = (x,y), xytext = (0, 0), textcoords = 'offset points', size = 1)    
    
    #plt.show()
    plt.savefig(output_file)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters for the classifier ')
    parser.add_argument('-i','--input', help='Embedding input file ', required=True)
    parser.add_argument('-o','--output', help='Output pdf file')
    parser.add_argument('-s','--size', help='Limit size of input for display', required=True)
    
    args = vars(parser.parse_args())
    input_file = args['input'];
    output_file = ""
    if args.has_key('output'):
        output_file = args['output'];
        
    size = 1000 
    if args.has_key('size'):
        size = int(args['size'])
     
    # Read the embedding file 
    model = load_embedding_file(input_file)
    print (" Size of model " + str(len(model)))
    
    # Visualize it (new) 
    visualize(output_file, model, size)
     

    
