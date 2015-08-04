package best_source;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import multiple_source.Multiple_source_fix_word_embedding;
import multiple_source.Uti;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Performance_each_language_pair implements Runnable{
	
	private String sourceLang = "";
	private String targetLang = "";
	private String outputFile = "";
	private String dataFile = "";
	private int iteration;
	private double adaGrad;
	
	public Performance_each_language_pair(double adaGrad, int iteration, String dataFile, String sourceLang, String targetLang, String outputFile){
		this.adaGrad  = adaGrad; 
		this.iteration = iteration ;
		this.dataFile = dataFile; 
		this.sourceLang = sourceLang;
		this.targetLang = targetLang;
		this.outputFile = outputFile; 
	}
	public void run() {

		try {
		Multiple_source_fix_word_embedding temp = new Multiple_source_fix_word_embedding();
		String embeddingFile = temp.train_crosslingual_word_embedding(dataFile, sourceLang, targetLang); 
        System.out.println("Embedding file : " + embeddingFile);
        // Default embedding Size = 50 
        
        String modelFile = temp.fix_the_word_train_parser(dataFile, sourceLang, targetLang, embeddingFile, adaGrad, iteration, 50); 
        System.out.println(" Model file : " + modelFile);
        
        System.out.println(" ==================EVALUATE PART ==========================");
        
        // Open the bufferedWriter 
        FileOutputStream fos = new FileOutputStream(this.outputFile);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));

        temp.EvaluateModel(dataFile, targetLang, modelFile, embeddingFile, bw);
        
        bw.close(); 
        fos.close();

		} 
		catch (Exception e){
			System.out.println(" ERROR FOR LANGUAGE PAIR : " + this.sourceLang + " : " + this.targetLang);
			System.out.println(e.getMessage());
		}

	}
	
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("AllLangs").withDescription("All languages needed to compute for pair").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("Adagrad").withDescription("Learning rate ").hasArg().withArgName("adagrad").create("adaGrad"));
        options.addOption(OptionBuilder.withLongOpt("Iteration").withDescription("Number of Iteration").hasArg().withArgName("iteration").create("iter"));
        options.addOption(OptionBuilder.withLongOpt("Thread").withDescription("Number of thread").isRequired().hasArg().withArgName("thread").create("thread"));
        
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
        String allLangs = commandLine.getOptionValue("AllLangs");
        String embeddingFile = "";
        int thread = Integer.parseInt(commandLine.getOptionValue("Thread"));
        
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
        if (!Uti.verifyLanguages(allLangs)){
        	throw new Exception(" Values of languages are not correct ");
        }
        
		ExecutorService es = Executors.newFixedThreadPool(thread);

        String[] langList = allLangs.split(",");
        for (int i =0; i<langList.length; i++){
        	for (int j =0; j<langList.length; j++){
        		String sourceLangs = langList[i];
        		String targetLangs = langList[j];
        		// Create a new thread and run the job 
        		Performance_each_language_pair temp = new Performance_each_language_pair(adaGrad,iteration,dataFile,
        				sourceLangs,targetLangs,"result/pair."+sourceLangs+"."+targetLangs);
    			es.execute(temp);
        	}
        }
		es.shutdown();
		es.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);		

        
        
	}
}
