__docformat__ = 'restructedtext en'

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from collections import OrderedDict

from utils import load_data, save_model, load_mapping_matrix, dropout_from_layer, save_embedding

class EmbedLayer(object):
    def __init__(self, input, E):
        """
        The embedding layer of a MLP: this does the mapping from each element of input to the corresponding embedding

        :type input: theano.tensor.dmatrix (int)
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type E: theano.tensor.dmatrix
        :param E: The embedding to apply for the input 
        """
        self.input = input
        self.E = E
        #idxs = T.imatrix()
        #x    = E[idxs].reshape((idxs.shape[0], idxs.shape[1]*E.shape[1]))
        #f = theano.function(inputs=[idxs], outputs=x)
        # This is the way to get real value from Theano 
        #self.output = f(input)
        self.output = self.E[self.input].reshape((input.shape[0], input.shape[1]* self.E.shape[1]))
        # parameters of the model
        self.params = [self.E]
        

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is cubic

        Hidden unit activation is given by: cubic(dot(input,W) + b)

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        """
        # Reinitialize it (don't care about other initialization) 
        #n_in = W.get_value().shape[0]
        #n_out = W.get_value().shape[1]
        if W is None:
            W_values = numpy.asarray(
                            rng.uniform(
                            low=-numpy.sqrt(6. / (n_in + n_out)),
                            high=numpy.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)
                            ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None : 
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
         
        self.input = input
        
        lin_output = T.dot(input,self.W) + self.b
        # Cubic function
        self.output = lin_output * lin_output * lin_output
        # parameters of the model
        self.params = [self.W, self.b]

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, dropout_rate,  n_in, n_out, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in = n_in, n_out = n_out, W=W, b=b)

        self.output = dropout_from_layer(rng, self.output, p=dropout_rate)
        
class SoftMaxLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, rng, input, n_in, n_out,  W = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type W: theano.tensor.TensorType
        :param W: the weight for this layer (note that we don't use any bias term) 

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        #n_in = W.get_value().shape[0]
        #n_out = W.get_value().shape[1]
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        self.W = W 
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W))
        # symbolic description of how to compute prediction as class whose

        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        # parameters of the model
        self.params = [self.W]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng,  input, refInput,  E, W1,B1, W2, refE, refW1, refB1, refW2, mapping, drop_out_rate = 0.5, drop_out_embedding_rate = 0.2):
        """Initialize the parameters for the multilayer perceptron

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type E: theano.tensor.TensorType
        :param E: the embedding file 

        """
        print (" Configuration Source ")
        n_in1 = W1.get_value().shape[1]
        n_out1 = W1.get_value().shape[0]        
        n_in2 = W2.get_value().shape[1]
        n_out2 = W2.get_value().shape[0]
        print ('W1 size ' + str(n_in1) +  ' * ' +   str(n_out1)) 
        print ('W2 size ' + str(n_in2) + ' * '  + str(n_out2)) 
        
        print (" ")
        
        print (" Configuration Reference ")
        ref_n_in1 = refW1.get_value().shape[1]
        ref_n_out1 = refW1.get_value().shape[0]        
        ref_n_in2 = refW2.get_value().shape[1]
        ref_n_out2 = refW2.get_value().shape[0]
        print ('W1 size ' + str(ref_n_in1) +  ' * ' +   str(ref_n_out1)) 
        print ('W2 size ' + str(ref_n_in2) + ' * '  + str(ref_n_out2)) 
        
        print (" Size of mapping matrix  : " )
        print mapping.get_value().shape
        
        # Creat the matrix of W1 n_in1 * n_out1 (2400 * 200) 
        ##########################
        # ORIGINAL MODEL #######
        self.embeddingLayer = EmbedLayer(input, E)
        dropout_embedding = dropout_from_layer(rng, self.embeddingLayer.output, p=drop_out_embedding_rate)
        self.dropout_HiddenLayer = DropoutHiddenLayer(rng, input = dropout_embedding, dropout_rate = drop_out_rate, 
                                                                                     n_in = n_in1, n_out = n_out1)

        self.hiddenLayer = HiddenLayer(rng, 
            input=self.embeddingLayer.output,
            n_in = n_in1,
            n_out = n_out1,
            W=self.dropout_HiddenLayer.W * (1- drop_out_embedding_rate),  # Currently don't dropout the Embedding layer 
            b=self.dropout_HiddenLayer.b)
        
        self.dropout_LogRegressionLayer = SoftMaxLayer(rng, 
            input=self.dropout_HiddenLayer.output,
            n_in = n_in2, 
            n_out = n_out2,
            W=None
        )  
        self.logRegressionLayer = SoftMaxLayer(rng, 
            input=self.hiddenLayer.output,
            n_in = n_in2, 
            n_out = n_out2,
            W= self.dropout_LogRegressionLayer.W * (1- drop_out_rate)
        )        
        
        # ##REFERENCE MODEL ####################
        ## SHARE THE WEIGHT WITH THE ORIGINAL MODEL ####
        
        
        self.refEmbeddingLayer = EmbedLayer(refInput, refE)
        self.ref_dropout_embedding = dropout_from_layer(rng, self.refEmbeddingLayer.output, p=drop_out_embedding_rate)
#        self.refHiddenLayer = HiddenLayer(rng,
        self.ref_dropout_HiddenLayer = DropoutHiddenLayer(rng, 
            input=self.ref_dropout_embedding,
            dropout_rate = drop_out_rate,
            n_in = ref_n_in1,
            n_out = ref_n_out1,
            W = self.dropout_HiddenLayer.W,
            b = self.dropout_HiddenLayer.b)

        self.ref_dropout_LogRegressionLayer = SoftMaxLayer(rng, 
            input=self.ref_dropout_HiddenLayer.output,
            n_in = ref_n_in2, 
            n_out = ref_n_out2,
            W = T.dot(self.dropout_LogRegressionLayer.W,mapping.T) 
        )
        
        # Normal regularlization 
        self.L2_sqr = ((self.embeddingLayer.E ** 2).sum() +  
                              (self.refEmbeddingLayer.E ** 2).sum() + 
                              (self.dropout_HiddenLayer.W **2).sum() + 
                              (self.dropout_LogRegressionLayer.W ** 2).sum()
                              )
        
        # Tieing together  (remember to transform the matrix 
#        self.reg_L2_sqr = (  
#                                   ((self.hiddenLayer.W - self.refHiddenLayer.W) ** 2).sum()
#                                   + ((self.hiddenLayer.b - self.refHiddenLayer.b) ** 2).sum()
#                                   + ((self.logRegressionLayer.W - T.dot(self.refLogRegressionLayer.W,mapping) )  ** 2).sum()  
#                                   )
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        
    
        self.negative_log_likelihood = (
            self.dropout_LogRegressionLayer.negative_log_likelihood
        )
        self.refNegative_log_likelihood = (
            self.ref_dropout_LogRegressionLayer.negative_log_likelihood
        )
        
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.params = (self.embeddingLayer.params +  self.dropout_HiddenLayer.params  + self.dropout_LogRegressionLayer.params + 
                       self.refEmbeddingLayer.params)
        
        # Initialize the params to hold the accumulate gradient of each params 
        self._accugrads =   [build_shared_zeros(t.shape.eval(),'accugrad') for t in self.params]
        
    
def test_mlp(learning_rate=0.01, L2_reg=0.00000001,  n_epochs=2000,
             dataset='theano.join.data', ref_dataset = 'ref.theano.join.data', batch_size=10000, max_iter = 5000, 
             output='theano.model.out', validation_freq = 100, ada_epsilon = 0.000001, alpha_share = 0.9, map_file = "labels.mapping", 
             english_file = ''):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path to the theano.classifier.data


   """
    print (" Learning with params : ")
    print (" Learning rate : " + str(learn_rate)); 
    print (" Regularlization params : " + str(L2_reg))
    print (" Alpha  of tieing together : " + str(alpha_share))
    print (" Batch size : "  + str(batch_size))
    print (" Max Iter : " + str(max_iter))
    print (" Evaluation frequency  : " + str(validation_freq))
    
    print ('... loading data ')
    
    ##### LOAD DATASET ORIGINAL and REF ##############
    print (' ----> load the mapping matrix ')
    mapping_matrix = load_mapping_matrix(map_file)
    
    print (' ----> load the original data ')
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    E = datasets[2]
    W1 = datasets[3]
    B1 = datasets[4]
    W2 = datasets[5]
    print (' ----> load the ref data ')
    ref_datasets = load_data(ref_dataset)
    ref_train_set_x, ref_train_set_y = ref_datasets[0]
    ref_valid_set_x, ref_valid_set_y = ref_datasets[1]
    refE = ref_datasets[2]
    refW1 = ref_datasets[3]
    refB1 = ref_datasets[4]
    refW2 = ref_datasets[5]

    # compute number of minibatches for training, validation and testing
    
    n_train_batches = train_set_x.owner.inputs[0].get_value(borrow=True).shape[0] / batch_size
    n_ref_train_batches = ref_train_set_x.owner.inputs[0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.owner.inputs[0].get_value(borrow=True).shape[0] / batch_size
    
    if train_set_x.owner.inputs[0].get_value(borrow=True).shape[0]  % batch_size > 100: n_train_batches +=1
    if valid_set_x.owner.inputs[0].get_value(borrow=True).shape[0] % batch_size > 100 : n_valid_batches +=1 
    
    print 'Training batches : ' + str(n_train_batches) 
    print 'Ref training batches : ' + str(n_ref_train_batches)
    print 'Valid batches : ' + str(n_valid_batches)
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    ref_index = T.lscalar() # Reference index to the source minibatch 
    
    x = T.imatrix('x')  # the data is presented as rasterized images
    xref = T.imatrix('xref')  # the data is presented as rasterized images
    yref = T.ivector('yref')  # the labels are presented as 1D vector of
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    rng = numpy.random.RandomState(1234)

    ###### DROP OUT RATE #############
    
    dropout_rate_hidden = 0.5
    dropout_rate_visible = 0.2
    #############################
        
    # construct the MLP class
    classifier = MLP(rng,
        input=x,
        refInput=xref,
        E=E, 
        W1=W1,
        B1=B1,
        W2 = W2,
        refE = refE,
        refW1 = refW1, 
        refB1 = refB1, 
        refW2 = refW2, 
        mapping = mapping_matrix,
        drop_out_rate=dropout_rate_hidden,
        drop_out_embedding_rate=dropout_rate_visible
    )

    train_errors = (classifier.errors(y))
    if (alpha_share > 1):
        raise ValueError(" Value of Alpha must be [0,1] ")
        
    cost = (
         alpha_share * classifier.negative_log_likelihood(y)
        + (1 - alpha_share) * classifier.refNegative_log_likelihood(yref)
        + L2_reg * classifier.L2_sqr
    )
    

    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch (remember index should always to even) 
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],   # x,y here is symbolic variable 
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            #xref: numpy.zeros(batch_size), 
            #yref: numpy.zeros(batch_size)
        }
    )

    # compute the gradient of cost with respect to theta 
    gparams = [T.grad(cost, param) for param in classifier.params]
    # Put the adagrad here 

    #learning_rate = T.fscalar('lr')  # learning rate to use
    updates = OrderedDict()
    for accugrad, param, gparam in zip(classifier._accugrads, classifier.params, gparams):
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + ada_epsilon)) * gparam
            updates[param] = param + dx
            updates[accugrad] = agrad

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index,ref_index],
        outputs=(cost, train_errors),
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],   # x,y here is symbolic variable 
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            xref: ref_train_set_x[ref_index * batch_size: (ref_index + 1) * batch_size],
            yref: ref_train_set_y[ref_index * batch_size: (ref_index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training '

    # early-stopping parameters
    patience = 2000  # Long Duong : At least have to went through this much iteration 
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    #validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    validation_frequency = validation_freq
    
    ######## FOR TESTING ONLY ##################
    #validation_frequency = 5 
    #n_train_batches = 10 
    #n_epochs = 1 
    ######################################
    
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    ref_batch_idx = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches): 
            (minibatch_avg_cost, minibatch_avg_error) = train_model(minibatch_index, ref_batch_idx)
            ref_batch_idx += 1
            if ref_batch_idx >= n_ref_train_batches:
                    ref_batch_idx = 0 
                
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            print (' Iteration :  ' + str(iter) + ' with Cost  (join) = ' + str(minibatch_avg_cost) + '  with errors (target only) = ' + str(minibatch_avg_error))
            # Long Duong : since in each epoch => n_train_batches has covered 
            # iter : is the number of update for the parameters (~ number of batches considered) 

            if (iter + 1) % validation_frequency == 0:
                # Note that because we 
                validation_losses = [validate_model( i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index ,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        # Long Duong : this is the key : need iter to get this good result => Waiting this much iter to expect 
                        # other better result ....  
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # Save the model  
                    save_model(output,classifier.embeddingLayer.E.get_value(), 
                                       (classifier.dropout_HiddenLayer.W.get_value() * (1-dropout_rate_visible )).T, 
                                       classifier.dropout_HiddenLayer.b.get_value(), 
                                       (classifier.dropout_LogRegressionLayer.W.get_value() * (1- dropout_rate_hidden)).T)
                    # Now save the nglish model
                    
                    if english_file is not None:
                        save_embedding(english_file,classifier.refEmbeddingLayer.E.get_value(),classifier.embeddingLayer.E.get_value())
                                            
            # Long Duong : add max_iter criterion 
            if (patience <= iter) or (iter > max_iter) :
                done_looping = True
                break

    
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i') %
          (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters for the classifier ')
    parser.add_argument('-l','--rate', help='Learning rate of the system', required=True, type=float)
    parser.add_argument('-r','--reg', help='regularlization params', required=True, type=float)
    parser.add_argument('-j','--alpha', help='alpha params for joinly train : alpha * X-ent (source) + (1-alpha) * X-ent(target)', required=True, type=float)
    parser.add_argument('-e','--eps', help='adagrad epsilon params', required=True, type=float)
    parser.add_argument('-b','--bsize', help='batch size (default 10000)', required=True, type=int)
    parser.add_argument('-i','--iter', help='iteration (default 5000)', required=True, type=int)
    parser.add_argument('-d','--input', help='Input dataset', required=True)
    parser.add_argument('-ref','--ref', help='Reference dataset', required=True)
    parser.add_argument('-o','--output', help='output model params', required=True)
    parser.add_argument('-f','--valfreq', help='validation frequency', required=True, type = int)
    parser.add_argument('-m','--mapping', help='Label mapping matrix (ref * original) ', required=True)
    parser.add_argument('-en','--eng', help='English Embedding Items files ', required=False)
    args = vars(parser.parse_args())
    # Args is the dictionary containing the argument
    learn_rate = args['rate'];
    reg_param = args['reg'];
    alpha_value= args['alpha'];
    eps_param = args['eps'];
    b_size = args['bsize'];
    iteration  = args['iter'];
    input_file = args['input'];
    ref_file = args['ref'];
    output_file = args['output']
    map_file = args['mapping']
    validation_freq = args['valfreq']
    english_file = '';
    if (args.has_key('eng')):
        english_file = args['eng']
        
    test_mlp(learning_rate = learn_rate, L2_reg = reg_param, n_epochs = 100000000, dataset = input_file, ref_dataset = ref_file,  
                  batch_size = b_size, max_iter = iteration, output=output_file, validation_freq = validation_freq, ada_epsilon = eps_param, alpha_share = alpha_value,
                  map_file = map_file, english_file = english_file)
    