package join_training;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
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

public class Extract_Translation  {
	
	private Map<Integer,String> srcVoc, trgVoc;
	private Map<String,Double> transTable; 
	private double THRESHOLD ;
	
	private void loadDictionary(String sourceVocFile, String targetVocFile ) throws IOException{
		srcVoc = new HashMap<Integer,String>(); 
		trgVoc = new HashMap<Integer, String>();
		
		FileInputStream fin = new FileInputStream(sourceVocFile); 
		BufferedReader br = new BufferedReader(new InputStreamReader( fin,"UTF8"));
		String line; 
		while ((line = br.readLine())!=null){
			StringTokenizer tokenizer = new StringTokenizer(line);
			Integer ID = Integer.parseInt(tokenizer.nextToken()); 
			String word = tokenizer.nextToken(); // Key is the string  
			srcVoc.put(ID, word);			
		}
		br.close();
		fin.close();
		///////////// Do the same with other Vocabulary file //////////////
		fin = new FileInputStream(targetVocFile); 
		br =  new BufferedReader(new InputStreamReader( fin,"UTF8"));
		while ((line = br.readLine())!=null){
			StringTokenizer tokenizer = new StringTokenizer(line);
			Integer ID = Integer.parseInt(tokenizer.nextToken()); 
			String word = tokenizer.nextToken(); // Key is the string  
			trgVoc.put(ID, word);						
		}
		
		br.close();
		fin.close();
		System.out.println(srcVoc.size());
		System.out.println(trgVoc.size());
	}
	private void output(String translationFile, String outputFile, double threshold) throws NumberFormatException, IOException{
		HashMap<String,String> saveTrans = new HashMap<String, String>();
		HashMap<String,Double> saveMax = new HashMap<String, Double>();
		
		FileOutputStream fos = new FileOutputStream(outputFile);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
		
		FileReader fr = new FileReader(translationFile); 
		BufferedReader br = new BufferedReader(fr);
		String line; 
		while ((line = br.readLine())!=null){
			StringTokenizer tokenizer = new StringTokenizer(line);
			int src_id = Integer.parseInt(tokenizer.nextToken().trim()); 
			int trg_id = Integer.parseInt(tokenizer.nextToken().trim());
			double prob = Double.parseDouble(tokenizer.nextToken().trim());
			String source = srcVoc.get(src_id);
			String target = trgVoc.get(trg_id);
			
			double currentMax =0 ; 
			if (saveMax.containsKey(source)) currentMax = saveMax.get(source);
			if (currentMax < prob){
				saveTrans.put(source, target);
				saveMax.put(source, prob);
			}			
			// if (prob >= threshold)
//			bw.write(String.format("%s %s %.10f \n", srcVoc.get(src_id) , trgVoc.get(trg_id), prob));
		}
		// Write the output 
		for (String key : saveTrans.keySet()){
			bw.write(key + " "  + saveTrans.get(key) + "\n");
		}
		
		br.close();
		fr.close();
		bw.close();
		fos.close();
		
	}

	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("SourceVoc").withDescription("Source Vocabulary File").isRequired().hasArg().withArgName("s").create("s"));        
        options.addOption(OptionBuilder.withLongOpt("TargetVoc").withDescription("Target Vocabulary File").isRequired().hasArg().withArgName("t").create("t"));
        options.addOption(OptionBuilder.withLongOpt("TranslationTable").withDescription("Translation Table file").isRequired().hasArg().withArgName("trans").create("trans"));
        options.addOption(OptionBuilder.withLongOpt("Threshold").withDescription("Threshold for translation table").isRequired().hasArg().withArgName("threshold").create("threshold"));
        options.addOption(OptionBuilder.withLongOpt("OutputFile").withDescription("Output File").isRequired().hasArg().withArgName("output").create("out"));
        
        
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

        String sourceVoc = commandLine.getOptionValue("SourceVoc");
        String targetVoc = commandLine.getOptionValue("TargetVoc");
        String transTable = commandLine.getOptionValue("TranslationTable");
        double threshold = Double.parseDouble(commandLine.getOptionValue("Threshold"));
        String outputFile = commandLine.getOptionValue("OutputFile");
        Extract_Translation temp = new Extract_Translation();
        temp.loadDictionary(sourceVoc, targetVoc);
        temp.output(transTable, outputFile, threshold);
	}
}
