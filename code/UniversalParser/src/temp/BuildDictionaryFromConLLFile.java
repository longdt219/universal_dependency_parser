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

public class BuildDictionaryFromConLLFile {
	public void readFile(String fileName, HashMap<String,Boolean> dict) throws IOException{
		FileInputStream sourceFile = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(sourceFile,"UTF8"));
		String line = "";
		int count = 0; 
		while ((line=br.readLine())!=null){
			count++; 
			line = line.trim(); 
			if (line.equals("")){
				continue; // without caring about this one 
			}
			String[] tokens = line.split("\\s+");  
			String word = tokens[1];
			if (!dict.containsKey(word)) dict.put(word, true);
		}
		br.close();
		sourceFile.close();
	}
	
	public void writeFile(String fileName, HashMap<String,Boolean> dict) throws IOException{
		FileOutputStream fos = new FileOutputStream(fileName);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF8"));
		for (String key : dict.keySet()){
			bw.write(key + "\n");
		}
		bw.close(); fos.close();
	}
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("InputTrain").withDescription("Path to the input ConLL Train file ").isRequired().hasArg().withArgName("inputTrain").create("itrain"));        
        options.addOption(OptionBuilder.withLongOpt("InputTest").withDescription("Path to the input ConLL Test file ").isRequired().hasArg().withArgName("inputTest").create("itest"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output dictionary file ").isRequired().hasArg().withArgName("output").create("o"));
        
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

        String fileNameTrain = commandLine.getOptionValue("InputTrain");
        String fileNameTest = commandLine.getOptionValue("InputTest");
        String output = commandLine.getOptionValue("Output");
        BuildDictionaryFromConLLFile temp = new BuildDictionaryFromConLLFile();
        
        HashMap<String, Boolean> dict = new HashMap<String, Boolean>(); 
		temp.readFile(fileNameTrain, dict);
		temp.readFile(fileNameTest, dict);
		temp.writeFile(output, dict);
	}
}
