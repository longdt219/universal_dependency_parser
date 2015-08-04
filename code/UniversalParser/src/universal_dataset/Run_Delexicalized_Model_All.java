package universal_dataset;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import multiple_source.Uti;
import multiple_source.invokeCMD;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class Run_Delexicalized_Model_All {
	public void readFile(String fileName, BufferedWriter bw, int windowSize) throws IOException{
		FileInputStream sourceFile = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(sourceFile,"UTF8"));
		String line = "";
		ArrayList<String> wordList = new ArrayList<String>();
		ArrayList<String> tagList = new ArrayList<String>();
		
		while ((line=br.readLine())!=null){
        	line=line.trim(); 
        	if (line.equals("")){
        		// Write to the file 
        		for (int i =0; i<wordList.size(); i++){
        			String suf = "";
        			String pref = "";
        			int start = i - windowSize; 
        			int end = i + windowSize;
        			if (i - windowSize < 0){
        				start = 0 ;
        				for (int k = (i- windowSize); k <0; k++){
        					suf += "UNKNOW"+k + " ";
        				}
        			}
        			if (i + windowSize >= wordList.size()){
        				end = wordList.size() - 1; 
        				for (int k = 1; k <=i + windowSize - wordList.size()+1; k++){
        					pref += "UNKNOW"+k + " ";
        				}        				
        			}
        			bw.write(suf);
        			for (int j =start; j<=end; j++){
        				if (j != i )
        					bw.write(tagList.get(j) +(j-i) + " ");
        				else
        					bw.write(wordList.get(i) + " ");
        			}
        			bw.write(pref + "\n");
        		}
        		wordList.clear(); 
        		tagList.clear();
        		continue;
        	}
        	String[] tokens = line.split("\\s+");
        	// Ok now modify the values 
        	String tag = tokens[4]; // The fine grain tag 
        	String word = tokens[1]  ;  // Word form
        	wordList.add(word);
        	tagList.add(tag);
        }
		br.close();
		sourceFile.close();
	}
	public String fix_the_word_train_parser(String Data, String sourceLangs, String targetLang, double adaGrad, int iteration) throws IOException{
		// Firstly join the file together 

		String[] sLangs = sourceLangs.split(",");
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		invokeCMD ivk = new invokeCMD();
		
		int identity = (int) (Math.random() * 100000000 + 1);
		String finalTrainFile = String.format("Train.Conll.Upos.All.%d", identity);
		String finalTestFile  = String.format("Test.Conll.Upos.All.%d", identity);
		for (String sLang : sLangs){
			String train_file = Uti.get_training_file(file_list, sLang);
			String test_file = Uti.get_testing_file(file_list, sLang);

			String cmd = String.format("cat %s >> %s", train_file, finalTrainFile);
			ivk.runSimpleCommand(cmd, true);
			cmd = String.format("cat %s >> %s", test_file, finalTestFile);
			ivk.runSimpleCommand(cmd, true);
		}

		//====== Run the training here ================
		String modelFile = "multiple.source." + identity;
		String cmd = String.format("java -mx100g -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/*   edu.stanford.nlp.parser.nndep.DependencyParser "
				+ "-trainFile %s  -devFile %s -embeddingSize 50  -model %s -maxIter %d -adaAlpha %f ", finalTrainFile, finalTestFile,  modelFile, iteration, adaGrad);
		ivk.runSimpleCommand(cmd, true);
		
		return modelFile;
	}
	
	
	public void EvaluateModel(String Data, String sourceLangs, String targetLang, String modelFile) throws IOException{
		// Get the target test file 
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		String testFile = Uti.get_testing_file(file_list, targetLang);
		String[] sLangs = sourceLangs.split(",");
		invokeCMD ivk = new invokeCMD();
		
		// Evaluate on Source language
		
		if (sLangs.length == 1){ // Exactly 1 source language => evaluate
			String sourceTestFile = Uti.get_testing_file(file_list, sLangs[0]);
			String cmd = String.format("java -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/*   edu.stanford.nlp.parser.nndep.DependencyParser "
					+ "-model %s -testFile %s ", modelFile, sourceTestFile);
			ivk.runSimpleCommand(cmd, true);
		}
 
		
		// Evaluate on Target language directly
		String cmd = String.format("java -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/*   edu.stanford.nlp.parser.nndep.DependencyParser "
				+ "-model %s -testFile %s ", modelFile, testFile);
		ivk.runSimpleCommand(cmd, true);
		
		
	}
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("TargetLang").withDescription("Target languages").isRequired().hasArg().withArgName("targetLang").create("tLang"));
        options.addOption(OptionBuilder.withLongOpt("SourceLangs").withDescription("Source languages ").isRequired().hasArg().withArgName("sourceLangs").create("sLangs"));
        options.addOption(OptionBuilder.withLongOpt("Embedding").withDescription("Supply with a pre-trained embedding. If this was provided, no embedding is trained ").hasArg().withArgName("embedding").create("e"));
        options.addOption(OptionBuilder.withLongOpt("Adagrad").withDescription("Learning rate ").hasArg().withArgName("adagrad").create("adaGrad"));
        options.addOption(OptionBuilder.withLongOpt("Iteration").withDescription("Number of Iteration").hasArg().withArgName("iteration").create("iter"));
        
        options.addOption("h", "help", false, "Print this message");

        CommandLine commandLine = null;
        
        try {
            commandLine = parser.parse(options, args); // if not enough parameters ....
            if (commandLine.hasOption("help")) {       // also if help is presented
                throw new ParseException("");
            }
        } catch (ParseException exp) {
            System.out.println();
            if (exp.getMessage().length() > 0) {
                System.out.println("ERR: " + exp.getMessage());
                System.out.println();
            }
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(400, "java -mx4g " + Thread.currentThread().getStackTrace()[1].getClassName(), "\n", options, "\n", true);
            System.out.println();
            System.exit(0);
        }

        String dataFile = commandLine.getOptionValue("Data");
        String sourceLangs = commandLine.getOptionValue("SourceLangs");
        String targetLang = commandLine.getOptionValue("TargetLang");
        String embeddingFile = "";
        double adaGrad = 0.01; // Value by default 
        int iteration = 5000;
        if (commandLine.hasOption("Embedding")){
        	embeddingFile = commandLine.getOptionValue("Embedding");
        }
        if (commandLine.hasOption("Adagrad")){
        	adaGrad = Double.parseDouble(commandLine.getOptionValue("Adagrad"));
        }

        if (commandLine.hasOption("Iteration")){
        	iteration = Integer.parseInt(commandLine.getOptionValue("Iteration"));
        }

        // Verify the source and target languages
        if (!Uti.verifyLanguages(sourceLangs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        if (!Uti.verifyLanguages(targetLang)){
        	throw new Exception(" Value of target languages is not correct ");
        }
        
        Run_Delexicalized_Model_All temp = new Run_Delexicalized_Model_All();
        //String modelFile = "model.en.dev.embed.fix.word50.upos.mix.noType"; 
        //String embeddingFile = "word.embedding.11732360.removed";
        
        // Only trained if no are provided 
        
        String modelFile = temp.fix_the_word_train_parser(dataFile, sourceLangs, targetLang, adaGrad, iteration); 
        System.out.println(" Model file : " + modelFile);
        System.out.println(" ==================EVALUATE PART ==========================");
        temp.EvaluateModel(dataFile, sourceLangs, targetLang, modelFile);
	}
}
