package join_training;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
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

public class TrimDictionary  {
	

	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Translation").withDescription("Translation file  English - Target ").isRequired().hasArg().withArgName("t").create("t"));        
        options.addOption(OptionBuilder.withLongOpt("Mono").withDescription("Monolingual File for Frequency Cutoff").isRequired().hasArg().withArgName("m").create("m"));
        options.addOption(OptionBuilder.withLongOpt("Top").withDescription("Top n-word").isRequired().hasArg().withArgName("n").create("n"));
        options.addOption(OptionBuilder.withLongOpt("OutputFile").withDescription("Output trimed translation file").isRequired().hasArg().withArgName("o").create("o"));
        
        
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

        String translationFile = commandLine.getOptionValue("Translation");
        String monoFile = commandLine.getOptionValue("Mono");
        int n  = Integer.parseInt(commandLine.getOptionValue("Top"));
        String outputFile = commandLine.getOptionValue("OutputFile");
        TrimDictionary temp = new TrimDictionary();
        HashMap<String, Integer> freq = temp.getFrequency(monoFile, n);
        
	}
	private HashMap<String, Integer> getFrequency(String monoFile, int n) throws  IOException {
		FileInputStream fis = new FileInputStream(monoFile);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = "";
		HashMap<String, Integer> result = new HashMap<String, Integer>();
		while ((line = br.readLine())!=null){
			String[] tokens = line.trim().split("\\s+");
			for (String token : tokens){
				int value = 0; 
				if (result.containsKey(token))
					value = result.get(token);
				result.put(token, value + 1);
			}
		}
		br.close();
		fis.close();
		
		// Sort the hash according to value and cut the first n
		return result ;
	}
}
