package best_source;

import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;

import multiple_source.Uti;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class Normalize_Data_Size_UniversalDep {
	
	private void data_normalize(String fileName, String extension, int size) throws FileNotFoundException, UnsupportedEncodingException, IOException{
		
		FileOutputStream fos = new FileOutputStream(fileName+extension); 
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
		
		
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = "";
		int countTk = 0; 
		int countSt = 0; 
		while ((line = br.readLine())!=null){
			line = line.trim();
			if (line.startsWith("#")) continue;
			
			if (line.equals("")) {
				countSt += 1;
				if (countTk > size) break;
				bw.write("\n");
				continue; 
			}
			countTk +=1;
			bw.write(line + "\n");
		}
		br.close();
		fis.close();
		bw.close();
		fos.close();
		
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of Languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("Size").withDescription("Number of tokens to cut-off ").isRequired().hasArg().withArgName("Size").create("s"));
        options.addOption(OptionBuilder.withLongOpt("FileExtension").withDescription("File extension ").isRequired().hasArg().withArgName("extension").create("e"));
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
        String Langs = commandLine.getOptionValue("Langs");
        int size = Integer.parseInt(commandLine.getOptionValue("Size"));
        String fileExtension = commandLine.getOptionValue("FileExtension");
        
        // Verify the source and target languages
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
 

        Normalize_Data_Size_UniversalDep temp = new Normalize_Data_Size_UniversalDep();
        String[] langList = Langs.split(",");
        for (int i =0; i< langList.length; i++){
    		ArrayList<String> file_list = Uti.read_training_path_file(dataFile);
    		String Train_file = Uti.get_training_file(file_list, langList[i]);

        	temp.data_normalize(Train_file, fileExtension , size);
        }
        
	}	
}
