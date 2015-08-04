package best_source;

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

public class Language_Similarity_Matrix_POS_WALS {
	

	public double getSourceTargetPairScore(String Data,String sourceLang, String targetLang) throws IOException{
		
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		String sTrain_file = Uti.get_training_file(file_list, sourceLang);
		String tTrain_file = Uti.get_training_file(file_list, targetLang);
		Find_Best_Source fbs = new Find_Best_Source();
		
		HashMap<String, Double> result1 = fbs.calculate_for_file(sTrain_file,2);
		HashMap<String, Double> result2 = fbs.calculate_for_file(tTrain_file,2);
		double bigram = fbs.Calculate_jenshen_Sharnon(result1, result2);

		result1 = fbs.calculate_for_file(sTrain_file,3);
		result2 = fbs.calculate_for_file(tTrain_file,3);
		double trigram = fbs.Calculate_jenshen_Sharnon(result1, result2);

		result1 = fbs.calculate_for_file(sTrain_file,4);
		result2 = fbs.calculate_for_file(tTrain_file,4);
		double forthgram = fbs.Calculate_jenshen_Sharnon(result1, result2);

		result1 = fbs.calculate_for_file(sTrain_file,5);
		result2 = fbs.calculate_for_file(tTrain_file,5);
		double fifthgram = fbs.Calculate_jenshen_Sharnon(result1, result2);
		
		result1 = fbs.calculate_for_file(sTrain_file,6);
		result2 = fbs.calculate_for_file(tTrain_file,6);
		double xixthgram = fbs.Calculate_jenshen_Sharnon(result1, result2);
		
		double finalNumber = bigram * fbs.DISTANCE_1 + trigram  * (fbs.DISTANCE_2 -fbs.DISTANCE_1) + forthgram * (fbs.DISTANCE_3 - fbs.DISTANCE_2) 
				                            + fifthgram * (fbs.DISTANCE_4 -fbs.DISTANCE_3) + xixthgram * (fbs.DISTANCE_REST - fbs.DISTANCE_4);		
		//double finalNumber = bigram  + trigram  + forthgram  + fifthgram + xixthgram ;		
		return finalNumber;
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of Languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("WALS").withDescription("Wals achive file ").isRequired().hasArg().withArgName("wals").create("wals"));
        options.addOption(OptionBuilder.withLongOpt("Features").withDescription("Features that needed to take into consideration").isRequired().hasArg().withArgName("features").create("f"));

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

        String featureFile = commandLine.getOptionValue("Features");
        String walsFile = commandLine.getOptionValue("WALS");
        
        // Verify the source and target languages
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        Language_Similarity_Matrix_WALS tmp = new Language_Similarity_Matrix_WALS();
        
        ArrayList<String> listFeatures = tmp.readFeatureList(featureFile);
        String[] langList = Langs.split(",");
        ArrayList<String> listLangs = new ArrayList<String>();
        for (String lang : langList) listLangs.add(lang);
        
        HashMap<String, HashMap<String, Integer>> langRepresentation = tmp.getLanguageVectorRepresentation(listLangs,listFeatures,walsFile);


        Language_Similarity_Matrix_POS_WALS temp = new Language_Similarity_Matrix_POS_WALS();
        for (int i =0; i< langList.length; i++){
        	System.out.print(langList[i] +":");
        	for (int j =0; j< langList.length; j++){
        		System.out.print(temp.getSourceTargetPairScore(dataFile,langList[i],langList[j]) + ":");
        	}
        	System.out.println();
        }
        
	}
}
