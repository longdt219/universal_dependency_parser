package join_training;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import multiple_source.Uti;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

import multiple_source.*;

public class Join_Training_With_BiDict_Learning_Curve  implements Runnable{

	private String train_data;
	private String dev_data; 
	private String embedding_file; 
	private String Lang, Size;
	private String test_data;
	private String refTrain;
	private String refEmbeding_file;
	private int batchSize;
	private String prefix;
	private int maxIter;
	private String dictFile;
	private double regJoin; 
	
	public Join_Training_With_BiDict_Learning_Curve(String train, String test, String dev,  String embed,  String refEmbed, String Lang, String Size,  
			String refTrain, int maxIter, int batchSize, String prefix, String dictFile, double regJoin){
		this.train_data = train;
		this.test_data = test; 
		this.dev_data = dev; 
		this.embedding_file = embed;
		this.refEmbeding_file = refEmbed;
		this.Lang = Lang; 
		this.Size = Size;
		this.refTrain = refTrain;
		this.maxIter = maxIter; 
		this.prefix = prefix; 
		this.batchSize = batchSize;
		this.dictFile  = dictFile;
		this.regJoin  = regJoin;
	}
	
	public void run() {
		// Run each supervised NN parser 
		int ID = (int) (1000000 * Math.random());
		String model_name = "model.bidict.join." + Lang +"." + ID;
		String out_file = this.prefix + "." + Lang +"." + Size ;
		String input_theano = "theano.bidict.join.data." + ID;
		String ref_input_theano  = "ref.theano.join.data." + ID;
		String output_theano = "theano.model.bidict.join." + Lang + "." + Size;
		String mapping_file = "mapping.bidict.en." + this.Lang + "." + ID ;
		invokeCMD ivk = new invokeCMD();
		// FIX THIS VALUE 
				
		String cmd = String.format("java -mx100g -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/* edu.stanford.nlp.parser.nndep.DependencyParserJoinTraining  "
				+ "-trainFile %s -devFile %s -testFile %s  -refTrainFile %s -model %s -embeddingSize 50 -maxIter %d -batchSize %d  -embedFile %s -refEmbedFile %s "
				+ "-numPreComputed 0 -inputTheano %s -refInputTheano %s -outputTheano %s -mappingFile %s -trainer noReg -alpha 0.5 -outPut %s -transtable %s -regJoin %.10f", train_data,dev_data,test_data, refTrain, 
				model_name,  this.maxIter, this.batchSize, embedding_file, this.refEmbeding_file, input_theano, ref_input_theano,output_theano, mapping_file, out_file, this.dictFile, this.regJoin);
		
		ivk.runSimpleCommand(cmd, true);
	}


	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("SourceLang").withDescription("Source Language Use For Join Training").isRequired().hasArg().withArgName("sLang").create("sLang"));
        options.addOption(OptionBuilder.withLongOpt("DataPoints").withDescription("Data point (in k) for running ").isRequired().hasArg().withArgName("range").create("range"));
        options.addOption(OptionBuilder.withLongOpt("EmbeddingDataFile").withDescription("Embedding Data File").isRequired().hasArg().withArgName("embedFile").create("embed"));
        options.addOption(OptionBuilder.withLongOpt("Thread").withDescription("Number of Threads").isRequired().hasArg().withArgName("thread").create("thread"));
        options.addOption(OptionBuilder.withLongOpt("MaxIter").withDescription("Max number of iteration").hasArg().withArgName("iter").create("iter"));
        options.addOption(OptionBuilder.withLongOpt("Prefix").withDescription("Output file prefix").hasArg().withArgName("prefix").create("prefix"));
        options.addOption(OptionBuilder.withLongOpt("Batch").withDescription("Batch Size").hasArg().withArgName("batch").create("batch"));
        options.addOption(OptionBuilder.withLongOpt("RegJoin").withDescription("Regularlization param for the Dictionary Part").hasArg().withArgName("regJoin").create("regJoin"));
        options.addOption(OptionBuilder.withLongOpt("BiDict").withDescription("Bilingual Dictionary Specific File").isRequired().hasArg().withArgName("bidict").create("bidict"));
        
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
        String Langs = commandLine.getOptionValue("Langs");
        String sLang = commandLine.getOptionValue("SourceLang");
        String dataPoints = commandLine.getOptionValue("DataPoints");
        String embeddingFile = commandLine.getOptionValue("EmbeddingDataFile");
        String bidictFile = commandLine.getOptionValue("BiDict");
        String prefix = "result.join" ;
        int maxIter = 5000 ;
        int batchSize = 10000;
        double regJoin = 0.0001;
        if (commandLine.hasOption("RegJoin"))
        	regJoin = Double.parseDouble(commandLine.getOptionValue("RegJoin"));
        
        if (commandLine.hasOption("Prefix"))
        	prefix = commandLine.getOptionValue("Prefix");
        if (commandLine.hasOption("MaxIter"))
        	maxIter = Integer.parseInt(commandLine.getOptionValue("MaxIter"));
        
        if (commandLine.hasOption("Batch"))
        	batchSize = Integer.parseInt(commandLine.getOptionValue("Batch"));
        
        int thread_no = Integer.parseInt(commandLine.getOptionValue("Thread"));
        
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of target languages are not correct ");
        }
        if (!Uti.verifyLanguages(sLang)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        ExecutorService es = Executors.newFixedThreadPool(thread_no);
        
        String[] dataList = dataPoints.split(",");
        String[] langList = Langs.split(",");

        ArrayList<String> file_list = Uti.read_training_path_file(dataFile);
        int i =0; 
        for (String data_size : dataList){
        	i ++; 
        	// Run with data_size 
        	// 1. First cut the whole data (already cut) during supervisedNN 
        	// cut_training_data(dataFile, Langs, data_size);
        	// 2. Run the supervised with all the languages (each supervised => one thread)
        	for (String lang : langList){
        		String ref_train = Uti.get_training_file(file_list, sLang);
        		String test_file = Uti.get_testing_file(file_list, lang);
        		String dev_file = Uti.get_dev_file(file_list, lang);
        		String train_file = Uti.get_training_file(file_list, lang);
        		String small_train_file = train_file + "." + data_size;
        		String embedding_file = Uti.get_embed_file(embeddingFile, lang);
        		String ref_embedding_file = Uti.get_embed_file(embeddingFile, sLang);
        		String bidict_file = Uti.get_bidict_file(bidictFile, lang); 
        		if (bidict_file.equals(""))
        			throw new Exception(" DICTIONARY FILE NOT FOUND FOR LANGUGAGE " + lang);
        		
        		Join_Training_With_BiDict_Learning_Curve temp = new Join_Training_With_BiDict_Learning_Curve(small_train_file,  test_file, dev_file,  embedding_file, ref_embedding_file, lang, data_size, ref_train, maxIter,  batchSize, prefix, bidict_file, regJoin);
                es.execute(temp);
        	}

        }
		es.shutdown();
		es.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);		
	}

}
