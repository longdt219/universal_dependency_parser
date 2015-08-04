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

public class ComputeDictionaryCoverage {
	public void readFile(String fileName, HashMap<String,String> dict) throws IOException{
		FileInputStream sourceFile = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(sourceFile,"UTF8"));
		String line = "";
		while ((line=br.readLine())!=null){
			line = line.trim(); 
			if (line.equals("")){
				continue; // without caring about this one 
			}
			String[] tokens = line.split("\\s+",2);   
			String word = tokens[0];
			if (!dict.containsKey(word)) dict.put(word, tokens[1]);
		}
		br.close();
		sourceFile.close();
	}
	
	public void writeResult(String fileName,  String outFile, HashMap<String,String> dict) throws IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF8"));
		
		FileOutputStream fos = new FileOutputStream(outFile);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF8"));
		
		String line = ""; 
		int count = 0;
		int countOK = 0; 
		while ((line = br.readLine())!=null){
			line = line.trim();
			count ++; 
			if (dict.containsKey(line)){
				countOK ++;
				// Output to file 
			} 
		}
		System.out.println("The coverage is  " + countOK / (1.0 * count));
		br.close(); fis.close();
		bw.close(); fos.close();
	}
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Input").withDescription("Input word list of both training and testing").isRequired().hasArg().withArgName("input").create("i"));        
        options.addOption(OptionBuilder.withLongOpt("Dict").withDescription("Input to the dictionary").isRequired().hasArg().withArgName("dict").create("d"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output the dictionary of file in input").isRequired().hasArg().withArgName("output").create("o"));
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
        String dictFile = commandLine.getOptionValue("Dict");
        String outFile = commandLine.getOptionValue("Output");
        
        ComputeDictionaryCoverage temp = new ComputeDictionaryCoverage();
        
        HashMap<String, String> dict = new HashMap<String, String>(); 
		temp.readFile(dictFile, dict);
		temp.writeResult(inputFile, outFile, dict);
		
	}
}
