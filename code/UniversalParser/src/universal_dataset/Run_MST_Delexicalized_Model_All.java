package universal_dataset;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import multiple_source.Uti;
import multiple_source.invokeCMD;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class Run_MST_Delexicalized_Model_All {
	
	public void EvaluateModel(String Data,  String targetLangs, String sourceModel) throws IOException{
		// Get the target test file 
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		String[] tLangs = targetLangs.split(",");
		invokeCMD ivk = new invokeCMD();
		
		// Evaluate on Source language
		
		for (String tLang : tLangs){
			String targetTestFile = Uti.get_testing_file(file_list, tLang);
			String cmd = String.format("java -mx100g -cp ../../tools/mstparser:../../tools/mstparser/lib/trove.jar:../../tools/mstparser/output/classes/  mstparser.DependencyParser "
					                                 + " model-name:%s test test-file:%s output-file:out.txt eval gold-file:%s", sourceModel, targetTestFile,targetTestFile);
			System.out.println(" EVALUATING FOR TARGET LANGUAGE : " + tLang);
			ivk.runSimpleCommand(cmd, true);			
		}		
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the upos delex data set ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("TargetLang").withDescription("Target languages for delexicalized parser").isRequired().hasArg().withArgName("targetLang").create("tLang"));
        options.addOption(OptionBuilder.withLongOpt("Model").withDescription("Source Model ").isRequired().hasArg().withArgName("model").create("m"));
        
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
        String sourceModel = commandLine.getOptionValue("Model");
        String targetLang = commandLine.getOptionValue("TargetLang");


        // Verify the source and target languages
        if (!Uti.verifyLanguages(targetLang)){
        	throw new Exception(" Value of target languages is not correct ");
        }
        
        Run_MST_Delexicalized_Model_All temp = new Run_MST_Delexicalized_Model_All();
        temp.EvaluateModel(dataFile, targetLang, sourceModel);
	}
}
