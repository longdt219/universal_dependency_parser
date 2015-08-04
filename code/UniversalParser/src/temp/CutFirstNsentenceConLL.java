package temp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class CutFirstNsentenceConLL {
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Input").withDescription("Path to the input ConLL file ").isRequired().hasArg().withArgName("input").create("i"));
        options.addOption(OptionBuilder.withLongOpt("No").withDescription("Number of sentence").isRequired().hasArg().withArgName("number").create("n"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output ConLL file with just n-sentence").isRequired().hasArg().withArgName("output").create("o"));
        
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
        String outfileName = commandLine.getOptionValue("Output");
        int noLines = Integer.parseInt(commandLine.getOptionValue("No"));

		FileOutputStream targetFile = new FileOutputStream(outfileName); 
		BufferedWriter bw = new BufferedWriter( new OutputStreamWriter(targetFile,"UTF8")); 
		
		FileInputStream sourceFile = new FileInputStream(inputFile); 
		BufferedReader br = new BufferedReader(new InputStreamReader(sourceFile,"UTF8"));
		String line = "" ;
		int count = 0 ;
		while ((line=br.readLine())!=null){
			line = line.trim(); 
			if (line.equals("")){
				 count ++;
			}
			bw.write(line+"\n");
			if (count >= noLines) break; 
		}
		br.close();
		sourceFile.close();
		bw.close(); 
		targetFile.close();
	}
}
