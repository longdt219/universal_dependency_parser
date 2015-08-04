package best_source;

import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
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

public class Strip_off_universal_dependency {
	
	private void printTokenSentence(String fileName) throws FileNotFoundException, UnsupportedEncodingException, IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = "";
		int countTk = 0; 
		int countSt = 0; 
		while ((line = br.readLine())!=null){
			if (line.startsWith("#")) continue; 
			if (line.equals("")) {
				countSt += 1;
				continue; 
			}
			countTk +=1; 
		}
		br.close();
		fis.close();
		DecimalFormat df = new DecimalFormat("##.#");
		
		System.out.print(df.format(countTk / 1000.0) + "/" + df.format(countSt / 1000.0)+ ":");
	}
	public void getStatistic(String Data,String Lang, String flag) throws IOException{
		
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		String fileName = "";
		if (flag.equals("TRAIN"))
			fileName = Uti.get_training_file(file_list, Lang);
		if (flag.equals("TEST"))
			fileName = Uti.get_testing_file(file_list, Lang);
		if (flag.equals("DEV"))
			fileName = Uti.get_dev_file(file_list, Lang);
		if (fileName.equals(""))
			System.out.println(" EXPECT ERROR !");
		printTokenSentence(fileName);
		
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of Languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        
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
        
        
        // Verify the source and target languages
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
 

        Strip_off_universal_dependency temp = new Strip_off_universal_dependency();
        String[] langList = Langs.split(",");
        for (int i =0; i< langList.length; i++){
        	temp.getStatistic(dataFile, langList[i], "TRAIN");

        }
    	System.out.println();
        for (int i =0; i< langList.length; i++){
        	temp.getStatistic(dataFile, langList[i], "DEV");
        	
        }
        System.out.println();
        for (int i =0; i< langList.length; i++){
        	temp.getStatistic(dataFile, langList[i], "TEST");
        	
        }
        System.out.println();
	}
}
