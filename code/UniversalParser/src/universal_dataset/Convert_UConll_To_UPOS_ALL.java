package universal_dataset;

import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
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

public class Convert_UConll_To_UPOS_ALL {
	
	public void convert(String Data,String Langs) throws IOException{
		// Firstly join the file together 

		String[] LangList = Langs.split(",");
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		
		for (String lang : LangList){
			String Train_file = Uti.get_training_file(file_list, lang);
			String Test_file = Uti.get_testing_file(file_list, lang);
			String Dev_file = Uti.get_dev_file(file_list, lang);
			temp.MapConLLUToUPos.Convert(Train_file, Train_file +".upos");
			temp.MapConLLUToUPos.Convert(Test_file, Test_file +".upos");
			temp.MapConLLUToUPos.Convert(Dev_file, Dev_file +".upos");
		}
		
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Languages needed to convert").isRequired().hasArg().withArgName("sLangs").create("sLangs"));
        
        
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
        	throw new Exception(" Values of languages are not correct ");
        }

  
        Convert_UConll_To_UPOS_ALL temp = new Convert_UConll_To_UPOS_ALL();
        temp.convert(dataFile, Langs);
	}
}
