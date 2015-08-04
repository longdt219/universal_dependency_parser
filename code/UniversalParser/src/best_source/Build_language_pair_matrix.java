package best_source;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.text.DecimalFormat;

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

public class Build_language_pair_matrix {
	
	public double get_score(String fileName, String evaluation){
		try {
		FileInputStream fis = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = ""; 
		while ((line = br.readLine()) !=null){
			if (line.contains(evaluation)){
				// Find the number 
				String[] tokens = line.split("=");
				return Double.parseDouble(tokens[1]);
			}
		}
		br.close();
		fis.close();

		} catch (IOException e) {
			return 0; // Mean that this file is not yet constructed  
		}
		return -1 ;  // Mean that some thing is going wrong, exception 
	}	
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("AllLangs").withDescription("All languages needed to compute for pair").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("Folder").withDescription("Folder that contains the pair result").isRequired().hasArg().withArgName("folder").create("f"));
        options.addOption(OptionBuilder.withLongOpt("Measurement").withDescription("Result Measurement (LAS/UAS) without Punct ").hasArg().withArgName("metrix").create("m"));
        
        
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

        String allLangs = commandLine.getOptionValue("AllLangs");
        String folderName = commandLine.getOptionValue("Folder");
        String measurement = ""; 
        if (commandLine.hasOption("Measurement")) 
        	measurement =  commandLine.getOptionValue("Measurement");
        
        // Verify the source and target languages
        if (!Uti.verifyLanguages(allLangs)){
        	throw new Exception(" Values of languages are not correct ");
        }
        
        Build_language_pair_matrix temp = new Build_language_pair_matrix();
        DecimalFormat df = new DecimalFormat("##.#");
        		
        String[] langList = allLangs.split(",");
        for (int i =0; i<langList.length; i++){
        	String sourceLang = langList[i];
        	System.out.print(sourceLang+":");
        	for (int j =0; j<langList.length; j++){
        		String targetLang = langList[j];
        		// Create a new thread and run the job
        		String fileName = folderName + "pair."+sourceLang  + "." + targetLang; 
        		double uas = temp.get_score(fileName, "UAS");
        		double las = temp.get_score(fileName, "LAS");
        		if (measurement.toUpperCase().equals("UAS"))
        			System.out.print(df.format(uas) +":");
        		else 
        			if (measurement.toUpperCase().equals("LAS"))
        				System.out.print(df.format(las) +":");
        			else
        				System.out.print("(" + df.format(uas) + "," + df.format(las)+ ")"+":");
        	}
        	System.out.println();
        }
        
	}
}
