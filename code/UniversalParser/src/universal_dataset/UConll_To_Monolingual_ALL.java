package universal_dataset;

import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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

public class UConll_To_Monolingual_ALL {
	
	public void convert(String Data,String Langs) throws IOException{
		// Firstly join the file together 

		String[] LangList = Langs.split(",");
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		
		for (String lang : LangList){
			String Train_file = Uti.get_training_file(file_list, lang);
			String Test_file = Uti.get_testing_file(file_list, lang);
			String Dev_file = Uti.get_dev_file(file_list, lang);
			
			String outputFile = Train_file.replace("train", "all") + ".mono";
			FileOutputStream fos = new FileOutputStream(outputFile);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
			
			temp.ConLL2Monolingual.readFile(Train_file, bw);
			temp.ConLL2Monolingual.readFile(Test_file, bw);
			temp.ConLL2Monolingual.readFile(Dev_file, bw);
			
			bw.close(); 
			fos.close();			
		}
		
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list (upos ) of universal data ").isRequired().hasArg().withArgName("data").create("data"));        
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

  
        UConll_To_Monolingual_ALL temp = new UConll_To_Monolingual_ALL();
        temp.convert(dataFile, Langs);
	}
}
