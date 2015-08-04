# 1. Jointly train the English-Frech parser (assumming that French is a resource poor langauge) 

# Parameters: 
# 	trainFile: French train treebank 
#       devFile  : French development treebank 
#       testFile : French test treebank 
#       refTrainFile: English train treebank 
#       model : output trained model 
#       embeddingSie: number of dimension for word embedding 
#       batchSize : batch size for training 
#       maxIter : maximum number of iteration 
#       embedFile : pretrained embedding file for French 
#       refEmbedFile : pretrained embedding file for English 
#       Other parameters: used for debugging only
# Output: 
#	Aside from the model, we also learn the cross-lingual word embedding which is en.fr.join.embedding 

java -mx100g -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/* edu.stanford.nlp.parser.nndep.DependencyParserJoinTraining  -trainFile ../../data/universal-dep/universal-dependencies-1.0/french/fr-ud-train.conllu.upos.3000 -devFile ../../data/universal-dep/universal-dependencies-1.0/french/fr-ud-dev.conllu.upos -testFile ../../data/universal-dep/universal-dependencies-1.0/french/fr-ud-test.conllu.upos   -refTrainFile ../../data/universal-dep/universal-dependencies-1.0/english/en-ud-train.conllu.upos -model model.fr.join.dev -embeddingSize 50 -batchSize 1000 -maxIter 10000 -embedFile ../Embedding/universal.embedding.french -refEmbedFile ../Embedding/europarl.universal.embedding.en -numPreComputed 0 -inputTheano theano.join.data -refInputTheano ref.theano.join.data -outputTheano theano.model.join -mappingFile mapping.en.fr -trainer noReg -alpha 0.5 -saveSource en.fr.join


# 2. Assume that some dictionary is available, incorporate dictionary to the model 
# Additional parameters: 
# 	transtable: translation table ~ translation dictionary (view the sample file for format)
#       regJoin   : regularlization sensitivity for biligual dictionary part   

java -mx100g -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/* edu.stanford.nlp.parser.nndep.DependencyParserJoinTraining  -trainFile ../../data/universal-dep/universal-dependencies-1.0/french/fr-ud-train.conllu.upos.1200000 -devFile ../../data/universal-dep/universal-dependencies-1.0/french/fr-ud-dev.conllu.upos -testFile ../../data/universal-dep/universal-dependencies-1.0/french/fr-ud-test.conllu.upos   -refTrainFile ../../data/universal-dep/universal-dependencies-1.0/english/en-ud-train.conllu.upos -model model.fr.join.dev -embeddingSize 50 -batchSize 1000 -maxIter 10000 -embedFile ../Embedding/universal.embedding.french -refEmbedFile ../Embedding/europarl.universal.embedding.en -numPreComputed 0 -inputTheano theano.join.data -refInputTheano ref.theano.join.data -outputTheano theano.model.join -mappingFile mapping.en.fr -trainer noReg -alpha 0.5 -saveSource en.fr.dict -transtable BilingualDict/en.fr.dict -regJoin 0.0001
