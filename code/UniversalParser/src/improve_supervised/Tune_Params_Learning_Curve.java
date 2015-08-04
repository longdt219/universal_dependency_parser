package improve_supervised;

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

public class Tune_Params_Learning_Curve  implements Runnable{

	private String train_data;
	private String dev_data; 
	private String embedding_file; 
	private String Lang, Size;
	private String refReg;
	private String refModel;
	private int lambda; 
	
	public Tune_Params_Learning_Curve(String train, String dev, String refReg, String embed, String Lang, String Size, String refModel, int lambda){
		this.train_data = train; 
		this.refReg = refReg; 
		this.dev_data = dev; 
		this.embedding_file = embed; 
		this.Lang = Lang; 
		this.Size = Size;
		this.refModel = refModel;
		this.lambda = lambda;
	}
	
	public void run() {
		// Run each supervised NN parser 
		int ID = (int) (1000000 * Math.random());
		String model_name = "model." + Lang +"." + ID;
		String out_file = "";
		if (this.lambda == 1)
			out_file = "result.tune.lambda1." + Lang +"." + Size +"." + refReg;
		else
			out_file = "result.tune.lambda2." + Lang +"." + Size +"." + refReg;
		// Model file : model.sv.dev
		invokeCMD ivk = new invokeCMD();
		// Note : fix the regularlization of POS and ARC
		String cmd = ""; 
		if (this.lambda == 1)
			cmd = String.format("java -mx100g -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/* edu.stanford.nlp.parser.nndep.DependencyParser "
					+ "-trainFile %s -devFile %s -testFile %s -model %s -embeddingSize 50 -maxIter 5000 -embedFile %s -outPut %s -refModel %s -refRegPOSARC 0.1 "
					+ "-refReg %s", train_data, dev_data, dev_data, model_name, embedding_file, out_file, refModel, refReg);
		else 
			cmd = String.format("java -mx100g -cp ../../code/CoreNLP/classes/:../../code/CoreNLP/lib/* edu.stanford.nlp.parser.nndep.DependencyParser "
					+ "-trainFile %s -devFile %s -testFile %s -model %s -embeddingSize 50 -maxIter 5000 -embedFile %s -outPut %s -refModel %s -refRegPOSARC %s "
					+ "-refReg 0.001", train_data, dev_data, dev_data, model_name, embedding_file, out_file, refModel, refReg);
			
		ivk.runSimpleCommand(cmd, true);
	}

	
	public static void cut_training_data(String Data, String langs, String size){
		invokeCMD ivk = new invokeCMD();
		String cmd = String.format("java -cp ../../code/UniversalParser/bin/:../../code/UniversalParser/lib/* best_source.Normalize_Data_Size_UniversalDep  "
				+ "-data training_data_path_universal_upos -Langs %s -e '.%s' -s %s", langs, size, size); 
		ivk.runSimpleCommand(cmd, true);
	}
	

	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("DataPoints").withDescription("Data point (in k) for running ").isRequired().hasArg().withArgName("range").create("range"));
        options.addOption(OptionBuilder.withLongOpt("RefReg").withDescription("Regularlization parameters ").isRequired().hasArg().withArgName("reg").create("reg"));
        options.addOption(OptionBuilder.withLongOpt("RefModel").withDescription("Reference model").isRequired().hasArg().withArgName("ref").create("refModel"));
        options.addOption(OptionBuilder.withLongOpt("EmbeddingDataFile").withDescription("Embedding Data File").isRequired().hasArg().withArgName("embedFile").create("embed"));
        options.addOption(OptionBuilder.withLongOpt("Thread").withDescription("Number of Threads").isRequired().hasArg().withArgName("thread").create("thread"));
        options.addOption(OptionBuilder.withLongOpt("Lambda").withDescription("Tunning Lambda (default = 1) ").hasArg().withArgName("lambda").create("lambda"));
        
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
        String dataPoints = commandLine.getOptionValue("DataPoints");
        String refRegStr = commandLine.getOptionValue("RefReg");
        String refModel = commandLine.getOptionValue("RefModel");
        String embeddingFile = commandLine.getOptionValue("EmbeddingDataFile");
        int thread_no = Integer.parseInt(commandLine.getOptionValue("Thread"));
        int embeddingSize = 50 ; 
        int iteration = 5000;
        int lambda = 1 ; 
        if (commandLine.hasOption("Lambda")){
        	lambda = Integer.parseInt(commandLine.getOptionValue("Lambda"));
        }
        
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        
        ExecutorService es = Executors.newFixedThreadPool(thread_no);
        
        String[] dataList = dataPoints.split(",");
        String[] langList = Langs.split(",");
        String[] refRegs = refRegStr.split(",");
        ArrayList<String> file_list = Uti.read_training_path_file(dataFile);
        for (String data_size : dataList){
        	// Run with data_size 
        	// 1. First cut the whole data (already cut) during supervisedNN 
        	// cut_training_data(dataFile, Langs, data_size);
        	// 2. Run the supervised with all the languages (each supervised => one thread)
        	for (String lang : langList){
        		for (String refReg : refRegs){
            		String test_file = Uti.get_testing_file(file_list, lang);
            		String dev_file = Uti.get_dev_file(file_list, lang);
            		String train_file = Uti.get_training_file(file_list, lang);
            		String small_train_file = train_file + "." + data_size;
            		String embedding_file = Uti.get_embed_file(embeddingFile, lang);
                    Tune_Params_Learning_Curve temp = new Tune_Params_Learning_Curve(small_train_file,  dev_file,  refReg, embedding_file, lang, data_size, refModel, lambda);
                    es.execute(temp);
        			
        		}
        	}

        }
		es.shutdown();
		es.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);		
	}

}
