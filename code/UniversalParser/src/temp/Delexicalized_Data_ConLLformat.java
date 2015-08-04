package temp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.commons.cli.*;

public class Delexicalized_Data_ConLLformat {
	
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Input").withDescription("Path to the input file which contains labeled/unlabeled lexicalized data").isRequired().hasArg().withArgName("input").create("i"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output delexicalized file").isRequired().hasArg().withArgName("output").create("o"));
        
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

        String inputFile = commandLine.getOptionValue("Input");
        String outputFile = commandLine.getOptionValue("Output");
        
        // Ok read the input file
        FileInputStream fis = new FileInputStream(inputFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF8"));
        FileOutputStream fos = new FileOutputStream(outputFile);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF8"));
        
        String line = "";
        while ((line = br.readLine())!=null){
        	line=line.trim(); 
        	if (line.equals("")){
        		bw.write("\n");
        		continue;
        	}
        	String[] tokens = line.split("\\s+");
        	// Ok now modify the values 
        	String tag = tokens[4]; // The fine grain tag 
        	tokens[1] = tag;  // Word form 
        	tokens[2] = tag;  // Lemma 
        	// Write it back to file 
        	for (String token:tokens){
        		bw.write(token+"\t");
        	}
        	bw.write("\n");
        }
        
        
        br.close();  fis.close();
        bw.close(); fos.close();
	}


}
