package temp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import multiple_source.Uti;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class Depepdency_Relation_Distance_Statistic {

	public HashMap<Integer,Integer> calculate_for_file(String fileName) throws IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = "";
		HashMap<Integer,Integer> result = new HashMap<Integer,Integer>();
		
		while ((line = br.readLine())!=null){
			line = line.trim();
			if (line.equals("")){
				continue; 
			}
			String[] tokens = line.split("\t");
			int dep = Integer.parseInt(tokens[6]);
			int currentPosition = Integer.parseInt(tokens[0]);
			//int key = Math.abs(currentPosition - dep);
			int key = currentPosition - dep;
			int value = 0; 
			if (result.containsKey(key)) value = result.get(key);
			result.put(key, value + 1);
		}

		br.close(); fis.close();
		return result; 
	}
	
	public void getStatistic(String Data,String setLangs) throws IOException{
		// Firstly join the file together 

		String[] sLangs = setLangs.split(",");
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		for (String sLang : sLangs){
			String train_file = Uti.get_training_file(file_list, sLang);
			String test_file = Uti.get_testing_file(file_list, sLang);
			
			HashMap<Integer, Integer> result1 = calculate_for_file(train_file);
			HashMap<Integer, Integer> result2 = calculate_for_file(test_file);
			 
			for (int key1 : result1.keySet()){
				if (result2.containsKey(key1)){
					result2.put(key1, result2.get(key1) + result1.get(key1));
				}
				else {
					result2.put(key1, result1.get(key1));
				}
			}
			HashMap<Integer, Integer> result  = result2 ;
			int total = 0; 
			for (int key : result.keySet()){
				total += result.get(key);
			}
			//System.out.println(" For " + sLang + " Length = -3 .. 3");
			for (int key = -4; key <=4; key++){
				int value = 0; 
				if (result.containsKey(key))
					value = result.get(key);
				System.out.print(value/ (1.0 *total) +",");
			}
			System.out.println();
		}

	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Language to calculate").isRequired().hasArg().withArgName("Langs").create("Langs"));
        
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
        
        Depepdency_Relation_Distance_Statistic temp = new Depepdency_Relation_Distance_Statistic();
        temp.getStatistic(dataFile, Langs);
	}
}
