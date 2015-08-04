from __builtin__ import ValueError

__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy 

import theano
import theano.tensor as T

def save_embedding(fileName,Es,Et):
    """
    save the learned embedding to file (English embedding)
    """
    print (" ...... Saving ENGLISH EMBEDDING to file " + fileName + ".embedding")
    # Read  the 
    fin = open(fileName,"rb")
    items = fin.readlines();
    fin.close()
    sizeEs, embedSize = Es.shape;
    sizeEt, embedSize = Et.shape;
    
    if len(items) != sizeEs + sizeEt:
        raise ValueError(" Items size and embedding size are not matched, expect errors ")
    
    # Write the model 
    fout = open(fileName + ".embedding",'wb')
    i = 0 
    for ei in Es:
        fout.write('en_'+ items[i].strip()  + '\t' + ' '.join(str(item) for item in ei) + '\n')
        i +=1 
         
    for ei in Et:
        fout.write('fr_' + items[i].strip() + '\t' + ' '.join(str(item) for item in ei) + '\n')
        i +=1 

    fout.close()
    
def save_model(fileName, E, W1, B1, W2):
    """
    save the learned model
    :type model_params: list 
    :param model_params:  array of params 
    """
    #E = model_params[0]
    #W1 = model_params[1]
    #B1 = model_params[2]
    #W2 = model_params[3]
    print (" ...... Saving the model !!!!")
    # Write the model 
    fout = open(fileName,'wb')
    # Write E 
    sizeE, embedSize = E.shape;
    fout.write(str(sizeE) + ' ' + str(embedSize)+ '\n')
    for ei in E:
        fout.write(' '.join(str(item) for item in ei) + '\n')
        
    # Write W1 
    sizeW1, widthW1 = W1.shape;
    fout.write(str(sizeW1) + ' ' + str(widthW1) + '\n')
    for wi in W1:
        fout.write(' '.join(str(item) for item in wi) + '\n')
    # Write B1
    sizeB1 = B1.shape[0]
    fout.write(str(sizeB1) +'\n')
    for bi in B1:
        fout.write(str(bi) + '\n')
         
    # Write W2
    sizeW2, widthW2 = W2.shape;
    fout.write(str(sizeW2) + ' ' + str(widthW2) + '\n')
    for wi in W2:
        fout.write(' '.join(str(item) for item in wi) + '\n')
 
    fout.close()
def load_translation_vector(dataset):
    """
    Load the translation vectors here. Dataset in the form of en-word, fr-word, en-id, fr-id 
    Return vector of en-id and fr-id 
    """
    fin = open(dataset,'rb')
       
    v_ref = []
    v = [] 
    for line in fin.readlines():
        line = line.strip()
        tokens = line.split()
        v_ref.append(int(tokens[2]))
        v.append(int(tokens[3]))    
    fin.close() 
    
    v_ref_shared =  theano.shared(value=numpy.asarray(v_ref, dtype=theano.config.floatX), name='v_ref', borrow=True)
    v_shared =  theano.shared(value=numpy.asarray(v, dtype=theano.config.floatX), name='v', borrow=True)
    # The shared variable must always be floatX to stored on GPU. However, 
    # We have to cast it back to int because we need to use index.
     
    return T.cast(v_ref_shared, 'int32'), T.cast(v_shared, 'int32')

def load_mapping_matrix(dataset):
    fin = open(dataset,'rb')
    # Load the training dataset
    nExample = int(fin.readline().split()[0])
    features = []
    for i in range(nExample):
        feature = fin.readline()
        features.append([int(featureI) for featureI in feature.split()])
        
    result= theano.shared(value=numpy.asarray(features, dtype=theano.config.floatX), name='mapping', borrow=True)
    fin.close()
    return result
    
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string (to the theano file) 
    :param dataset: the path to the dataset 
    '''

    #############
    # LOAD DATA #
    #############
    #train_set, valid_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    fin = open(dataset,'rb')
    # Load the training dataset
    nExample = int(fin.readline())
    features = []
    labels = [] 
    for i in range(nExample):
        index = int(fin.readline())
        if index != i: 
            raise ValueError(" Some thing wrong here, index not matched (train dataset)")
        feature = fin.readline()
        features.append([int(featureI) for featureI in feature.split()])
        labels.append(fin.readline().split().index('1'))
        
    train_set = (numpy.array(features), numpy.array(labels))
    
    # Load the dev dataset 
    nExample = int(fin.readline())
    dev_features = []
    dev_labels = [] 
    for i in range(nExample):
        index = int(fin.readline())
        if index != i: 
            raise ValueError(" Some thing wrong here, index not matched (dev dataset)")
        feature = fin.readline()
        dev_features.append([int(featureI) for featureI in feature.split()])
        dev_labels.append(fin.readline().split().index('1'))
    valid_set = (numpy.array(dev_features), numpy.array(dev_labels))

    # Load Embedding 
    sizeE = int(fin.readline().split()[0])
    E_arr = [] 
    for i in range(sizeE):
        E_arr.append([float(featureI) for featureI in fin.readline().split()])
    
    E =  theano.shared(value=numpy.asarray(E_arr, dtype=theano.config.floatX), name='E', borrow=True)

    # Load W1
    sizeW1 = int(fin.readline().split()[0])
    W1_arr = [] 
    for i in range(sizeW1):
        W1_arr.append([float(featureI) for featureI in fin.readline().split()])
    W1 = theano.shared(value=numpy.asarray(W1_arr, dtype=theano.config.floatX), name='W1', borrow=True)
    
    # Load b 
    sizeB  = int(fin.readline())
    b_arr = []
    for i in range(sizeB):
        b_arr.append(float(fin.readline()))
    B1 = theano.shared(value=numpy.asarray(b_arr, dtype=theano.config.floatX), name='B1', borrow=True)
    
    # Load W2 
    sizeW2 = int(fin.readline().split()[0])
    W2_arr = [] 
    for i in range(sizeW2):
        W2_arr.append([float(featureI) for featureI in fin.readline().split()])
    W2 = theano.shared(value=numpy.asarray(W2_arr, dtype=theano.config.floatX), name='W2', borrow=True)
    
    fin.close()
    # Print to test ....
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return T.cast(shared_x,'int32'), T.cast(shared_y, 'int32')

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), E, W1, B1, W2]
    return rval


def load_data_norm(dataset):
    ''' Loads the dataset

    :type dataset: string (to the theano file) 
    :param dataset: the path to the dataset 
    '''

    #############
    # LOAD DATA #
    #############
    #train_set, valid_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    fin = open(dataset,'rb')
    # Load the training dataset
    nExample = int(fin.readline())
    features = []
    labels = [] 
    for i in range(nExample):
        index = int(fin.readline())
        if index != i: 
            raise ValueError(" Some thing wrong here, index not matched (train dataset)")
        feature = fin.readline()
        features.append([int(featureI) for featureI in feature.split()])
        labels.append(fin.readline().split().index('1'))
        
    train_set = (numpy.array(features), numpy.array(labels))
    
    # Load the dev dataset 
    nExample = int(fin.readline())
    dev_features = []
    dev_labels = [] 
    for i in range(nExample):
        index = int(fin.readline())
        if index != i: 
            raise ValueError(" Some thing wrong here, index not matched (dev dataset)")
        feature = fin.readline()
        dev_features.append([int(featureI) for featureI in feature.split()])
        dev_labels.append(fin.readline().split().index('1'))
    valid_set = (numpy.array(dev_features), numpy.array(dev_labels))

    # Load Embedding 
    sizeE = int(fin.readline().split()[0])
    E_arr = [] 
    for i in range(sizeE):
        E_arr.append([float(featureI) for featureI in fin.readline().split()])
    
    E =  theano.shared(value=numpy.asarray(E_arr, dtype=theano.config.floatX), name='E', borrow=True)

    # Load W1
    sizeW1 = int(fin.readline().split()[0])
    W1_arr = [] 
    for i in range(sizeW1):
        W1_arr.append([float(featureI) for featureI in fin.readline().split()])
    W1 = theano.shared(value=numpy.asarray(W1_arr, dtype=theano.config.floatX), name='W1', borrow=True)
    
    # Load b 
    sizeB  = int(fin.readline())
    b_arr = []
    for i in range(sizeB):
        b_arr.append(float(fin.readline()))
    B1 = theano.shared(value=numpy.asarray(b_arr, dtype=theano.config.floatX), name='B1', borrow=True)
    
    # Load W2 
    sizeW2 = int(fin.readline().split()[0])
    W2_arr = [] 
    for i in range(sizeW2):
        W2_arr.append([float(featureI) for featureI in fin.readline().split()])
    W2 = theano.shared(value=numpy.asarray(W2_arr, dtype=theano.config.floatX), name='W2', borrow=True)
    
    fin.close()
    # Print to test ....
    
    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return T.cast(shared_x,'int32'), T.cast(shared_y, 'int32')

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), E, W1, B1, W2]
    return rval

def dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


if __name__ == '__main__':
    load_data("theano.classifier.data")
