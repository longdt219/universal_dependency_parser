package neural.net;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class ConvertConLL_WORD_POS_mixture {
	public void readFile(String fileName, BufferedWriter bw, int windowSize) throws IOException{
		FileInputStream sourceFile = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(sourceFile,"UTF8"));
		String line = "";
		ArrayList<String> wordList = new ArrayList<String>();
		ArrayList<String> tagList = new ArrayList<String>();
		
		while ((line=br.readLine())!=null){
			if (line.startsWith("#")) continue; 
        	line=line.trim(); 
        	if (line.equals("")){
        		// Write to the file 
        		for (int i =0; i<wordList.size(); i++){
        			String suf = "";
        			String pref = "";
        			int start = i - windowSize; 
        			int end = i + windowSize;
        			if (i - windowSize < 0){
        				start = 0 ;
        				for (int k = (i- windowSize); k <0; k++){
        					suf += "UNKNOW"+k + " ";
        				}
        			}
        			if (i + windowSize >= wordList.size()){
        				end = wordList.size() - 1; 
        				for (int k = 1; k <=i + windowSize - wordList.size()+1; k++){
        					pref += "UNKNOW"+k + " ";
        				}        				
        			}
        			bw.write(suf);
        			for (int j =start; j<=end; j++){
        				if (j != i )
        					//bw.write(tagList.get(j) +(j-i) + " ");
        					bw.write(tagList.get(j)  + " ");
        				else
        					//bw.write(wordList.get(i) + " ");
        					bw.write(wordList.get(i) + " ");
        			}
        			bw.write(pref + "\n");
        		}
        		wordList.clear(); 
        		tagList.clear();
        		continue;
        	}
        	String[] tokens = line.split("\\s+");
        	// Ok now modify the values 
        	String tag = tokens[3]; // The fine grain tag 
        	String word = tokens[1]  ;  // Word form
        	wordList.add(word);
        	tagList.add(tag);
        }
		br.close();
		sourceFile.close();
	}
	
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("InputTrain").withDescription("Path to the input ConLL Train file ").isRequired().hasArg().withArgName("inputTrain").create("itrain"));        
        options.addOption(OptionBuilder.withLongOpt("InputTest").withDescription("Path to the input ConLL Test file ").isRequired().hasArg().withArgName("inputTest").create("itest"));
        options.addOption(OptionBuilder.withLongOpt("WindowSize").withDescription("Windows Size (-z to z) ").isRequired().hasArg().withArgName("windowSize").create("wSize"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output mixture file ").isRequired().hasArg().withArgName("output").create("o"));
        
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
        int windowSize = Integer.parseInt(commandLine.getOptionValue("WindowSize"));
        String output = commandLine.getOptionValue("Output");
        ConvertConLL_WORD_POS_mixture temp = new ConvertConLL_WORD_POS_mixture();
        
        FileOutputStream fos = new FileOutputStream(output);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
        
		temp.readFile(fileNameTrain, bw, windowSize);
		temp.readFile(fileNameTest, bw, windowSize);
		
		bw.close();
		fos.close();
	
	}
}
