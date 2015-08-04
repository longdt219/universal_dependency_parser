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

public class MapConLLToUPos {
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Input").withDescription("Path to the input ConLL file ").isRequired().hasArg().withArgName("input").create("i"));
        options.addOption(OptionBuilder.withLongOpt("Mapping").withDescription("Mapping tagset file").isRequired().hasArg().withArgName("mapping").create("m"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output ConLL file with UPOS").isRequired().hasArg().withArgName("output").create("o"));
        
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

        String fileName = commandLine.getOptionValue("Input");
        String outfileName = commandLine.getOptionValue("Output");
        String mappingFile = commandLine.getOptionValue("Mapping");

        // Load the mapping 
        
		HashMap<String,String> mapping = new HashMap<String,String>();
		FileInputStream mapFile = new FileInputStream(mappingFile); 
		BufferedReader mapBr = new BufferedReader(new InputStreamReader(mapFile,"UTF8"));
		String line ; 
		while ((line=mapBr.readLine())!=null){
			String[] tokens = line.split("\t"); 
			String firstTag = tokens[0];
			String secondTag = tokens[1];
			mapping.put(firstTag, secondTag);
		}
		mapBr.close();
		mapFile.close();
		
		
		
		FileOutputStream targetFile = new FileOutputStream(outfileName); 
		BufferedWriter bw = new BufferedWriter( new OutputStreamWriter(targetFile,"UTF8")); 
		
		FileInputStream sourceFile = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(sourceFile,"UTF8"));
	
		while ((line=br.readLine())!=null){
			line = line.trim(); 
			if (line.equals("")){
				bw.write("\n");// Empty line
				continue; // without caring about this one 
			}
			String[] tokens = line.split("\\s");  
			  
			String tag = tokens[4];
			// handle the case when word is combined
			String mapTagAll = "NOUN";
			if (mapping.containsKey(tag)){
				mapTagAll = mapping.get(tag);
			}
			else 
				if (tag.contains("_")){
					String[] tagTks = tag.split("_");
					for (int i=0; i<tagTks.length; i++){
						if (mapping.containsKey(tagTks[i])){
							mapTagAll =  mapping.get(tagTks[i]);
						}
						else {
							System.out.println("(join tag) Not match with tag " + tagTks[i] + " from word = " + tokens[1]);
						}
					}
				}
				else {
						System.out.println("(single tag) Not match with tag " + tag + " from word = " + tokens[1]);
				}
			
			// Write to output file both general and specific tag have the same value of universal tag
			tokens[3] = mapTagAll;
			tokens[4] = mapTagAll; 
			tokens[5] = "_"; // Remove this information 
			tokens[8] = "_"; // Remove this information
			tokens[9] = "_"; // Remove this information
			for (String token:tokens)
				bw.write(token+"\t");
			bw.write("\n");
		}
		br.close();
		sourceFile.close();
		bw.close(); 
		targetFile.close();
	}
}
