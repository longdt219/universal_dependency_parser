package best_source;

import java.awt.List;

import org.apache.commons.csv.*; 

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
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

public class Language_Similarity_Matrix_WALS {
	

	public double cosineSimilarity(HashMap<String, Integer> sourceVector, HashMap<String, Integer> targetVector) throws IOException{
		// Return a cosine similarity between 2 distribution  
		double sizeA = Math.sqrt(sourceVector.size()); 
		double sizeB = Math.sqrt(targetVector.size());
		int count = 0;
		for (String key  : sourceVector.keySet()){
			if (targetVector.containsKey(key)){
				count ++ ;
			}
		}
		return count / (sizeA * sizeB);
	}

	public ArrayList<String> readFeatureList(String featureFile) throws IOException {
		ArrayList<String> result = new ArrayList<String>(); 
		File csvData = new File(featureFile);
		CSVParser parser = CSVParser.parse(csvData, Charset.forName("UTF-8"), CSVFormat.EXCEL);
		int count = 0; 
		for (CSVRecord csvRecord : parser) {
			 count ++; 
			 if (count ==1) continue; 
		     String featureID = csvRecord.get(3);
		     result.add(featureID);
		 }
		 return result; 
	}

	public HashMap<String,HashMap<String,Integer>> getLanguageVectorRepresentation(ArrayList<String> listLangs,
			ArrayList<String> listFeatures, String walsFile) throws IOException {
		
		HashMap<String,HashMap<String,Integer>> result = new HashMap<String, HashMap<String,Integer>>();
		
		File csvData = new File(walsFile);
		CSVParser parser = CSVParser.parse(csvData, Charset.forName("UTF-8"), CSVFormat.EXCEL);
		int count = 0;
		ArrayList<Integer> idx_features = new ArrayList<Integer>();
		for (CSVRecord csvRecord : parser) {
			 count ++; 
			 if (count ==1) {
				 // Header 
				 for (int i=0; i<csvRecord.size(); i++){
					 String header = csvRecord.get(i);
					 String featureName = header.split("\\s+",2)[0]; // eg. 87A
					 if (listFeatures.contains(featureName)){
						 idx_features.add(i);
					 }
				 }
				 if (idx_features.size() != listFeatures.size())
					 System.out.println(" EXPECT ERROR SINCE CAN'T FIND ALL FEATURES " + idx_features.size() + ".." + listFeatures.size());
			 }
			 
		     String langName = csvRecord.get(3).trim().toLowerCase(); //e.g: Spanish 
		     if (listLangs.contains(langName)){
		    	 HashMap<String, Integer> langVector = new HashMap<String, Integer>();
		    	 for (int idx : idx_features){
		    		 String featureValue = csvRecord.get(idx);
		    		 langVector.put("FEATURE" + idx + "_" + featureValue, 1);
		    	 } 
		    	 result.put(langName, langVector);
		     }
		 }
		
		if (result.size() != listLangs.size()){
			System.out.println(" Not all the languages are covered");
			return null;
		}else
			return result; 
		
	}
	
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        //options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
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

        String featureFile = commandLine.getOptionValue("Features");
        String walsFile = commandLine.getOptionValue("WALS");
        String Langs = commandLine.getOptionValue("Langs");
        
        
        // Verify the source and target languages
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        
        Language_Similarity_Matrix_WALS temp = new Language_Similarity_Matrix_WALS();
        ArrayList<String> listFeatures = temp.readFeatureList(featureFile);
        String[] langList = Langs.split(",");
        ArrayList<String> listLangs = new ArrayList<String>();
        for (String lang : langList) listLangs.add(lang);
        HashMap<String, HashMap<String, Integer>> langRepresentation = temp.getLanguageVectorRepresentation(listLangs,listFeatures,walsFile);
        
        
        
        for (int i =0; i< listLangs.size(); i++){
        	System.out.print(listLangs.get(i) +":");
        	for (int j =0; j< listLangs.size(); j++){
        		HashMap<String, Integer> sourceVector = langRepresentation.get(listLangs.get(i));
        		HashMap<String, Integer> targetVector = langRepresentation.get(listLangs.get(j)); 
        		System.out.print(temp.cosineSimilarity(sourceVector, targetVector) + ":");
        	}
        	System.out.println();
        }
        
	}


}
