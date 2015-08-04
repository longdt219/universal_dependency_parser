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

public class Find_Best_Source {
	
	public final double DISTANCE_1 = 0.50582931;
	public final double DISTANCE_2 = 0.69664809;
	public final double DISTANCE_3 = 0.78197317;
	public final double DISTANCE_4 = 0.83134353;
	public final double DISTANCE_REST = 1;
	
	public HashMap<String,Double> calculate_for_file(String fileName, int ngram) throws IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = "";
		HashMap<String,Double> result = new HashMap<String,Double>();
		HashMap<String, Integer> countNgram = new HashMap<String, Integer>();
		ArrayList<String> sentence = new ArrayList<String>();
		while ((line = br.readLine())!=null){
			line = line.trim();
			if (line.equals("")){
				// Calculate the statistic about n-gram 
				for (int i =0; i<sentence.size() - ngram + 1; i++){
					String key = ""; 
					for (int j = 0; j<ngram; j++)
						key += sentence.get(i +j) + "_" ;
					int value = 0; 
					if (countNgram.containsKey(key)){
						value = countNgram.get(key);
					}
					countNgram.put(key, value + 1);
				}
				
				sentence.clear();
				continue; 
			}
			String[] tokens = line.split("\t");
			String pos = tokens[4];
			sentence.add(pos);
		}

		br.close(); fis.close();
		// Now convert to probability
		double total = 0.0; 
		for (String key : countNgram.keySet()){
			total += countNgram.get(key);
		}
		
		for (String key : countNgram.keySet()){
			result.put(key, countNgram.get(key)/ (1.0 * total));
		}
		return result; 
	}
	
	public double Calculate_jenshen_Sharnon(HashMap<String, Double> x, HashMap<String, Double> y){
		// Convert x and y to equal length array
		ArrayList<Double> x1 = new ArrayList<Double>();
		ArrayList<Double> y1 = new ArrayList<Double>();
		for (String keyx : x.keySet()){
			x1.add(x.get(keyx));
			if (y.containsKey(keyx)){
				y1.add(y.get(keyx));
			}
			else {
				y1.add(0.0);
			}
		}
		for (String keyy : y.keySet()){
			if (!x.containsKey(keyy)){
				y1.add(y.get(keyy));
				x1.add(0.0);
			}
		}
		double[] x2 = new double[x1.size()];
		for (int i =0; i< x1.size(); i++){
			x2[i] = x1.get(i);
		}

		double[] y2 = new double[y1.size()];
		for (int i =0; i< y1.size(); i++){
			y2[i] = y1.get(i);
		}

		
		return Uti.jensenShannonDivergence(x2,y2);
	}
//	private static Map<String, Double> sortByComparator(Map<String, Double> unsortMap) {
//		 
//		// Convert Map to List
//		LinkedList<Map.Entry<String, Double>> list =  new LinkedList<Map.Entry<String, Double>>(unsortMap.entrySet());
// 
//		// Sort list with comparator, to compare the Map values
//		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
//			public int compare(Map.Entry<String, Double> o1,
//                                           Map.Entry<String, Double> o2) {
//				return (o1.getValue()).compareTo(o2.getValue());
//			}
//		});
// 
//		// Convert sorted map back to a Map
//		Map<String, Double> sortedMap = new LinkedHashMap<String, Double>();
//		for (Iterator<Entry<String, Double>> it = list.iterator(); it.hasNext();) {
//			Map.Entry<String, Double> entry = it.next();
//			sortedMap.put(entry.getKey(), entry.getValue());
//		}
//		return sortedMap;
//	}
	public void getBestSourceLang(String Data,String sourceLangs, String targetLang) throws IOException{
		// Firstly join the file together 

		String[] sLangs = sourceLangs.split(",");
		ArrayList<String> file_list = Uti.read_training_path_file(Data);
		double max = 0; 
		String bestSource = "";
		HashMap<String, Double> result = new HashMap<String, Double>();
		for (String sLang : sLangs){
			String sTrain_file = Uti.get_training_file(file_list, sLang);
			String tTrain_file = Uti.get_training_file(file_list, targetLang);
			
			HashMap<String, Double> result1 = calculate_for_file(sTrain_file,2);
			HashMap<String, Double> result2 = calculate_for_file(tTrain_file,2);
			double bigram = Calculate_jenshen_Sharnon(result1, result2);

			result1 = calculate_for_file(sTrain_file,3);
			result2 = calculate_for_file(tTrain_file,3);
			double trigram = Calculate_jenshen_Sharnon(result1, result2);

			result1 = calculate_for_file(sTrain_file,4);
			result2 = calculate_for_file(tTrain_file,4);
			double forthgram = Calculate_jenshen_Sharnon(result1, result2);

			result1 = calculate_for_file(sTrain_file,5);
			result2 = calculate_for_file(tTrain_file,5);
			double fifthgram = Calculate_jenshen_Sharnon(result1, result2);
			
			result1 = calculate_for_file(sTrain_file,6);
			result2 = calculate_for_file(tTrain_file,6);
			double xixthgram = Calculate_jenshen_Sharnon(result1, result2);
			
			double finalNumber = bigram * DISTANCE_1 + trigram  * DISTANCE_2 + forthgram * DISTANCE_3 
					                            + fifthgram * DISTANCE_4 + xixthgram * DISTANCE_REST;
			
			System.out.println(" Source Language = " + sLang + " Values of each elements : " + bigram + " " + trigram + "  " + forthgram +" " + fifthgram);
			result.put(sLang, finalNumber);
		}
		//Map<String, Double> temp = sortByComparator(result);
		Map<String, Double> temp = result;
		System.out.println(" Sorted List of source languages for target language : " + targetLang);
		for (Map.Entry<String, Double> entry : temp.entrySet()) {
			System.out.println("[Lang] : " + entry.getKey() 
                                      + " [JS Divergence] : " + entry.getValue());
		}
		
		
	}
	
	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("SourceLangs").withDescription("Source Languages to choose from").isRequired().hasArg().withArgName("sLangs").create("sLangs"));
        options.addOption(OptionBuilder.withLongOpt("TargetLang").withDescription("Target Language").isRequired().hasArg().withArgName("tLang").create("tLang"));
        
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
        String sLangs = commandLine.getOptionValue("SourceLangs");
        String tLang = commandLine.getOptionValue("TargetLang");
        
        // Verify the source and target languages
        if (!Uti.verifyLanguages(sLangs)){
        	throw new Exception(" Values of source languages are not correct ");
        }

        if (!Uti.verifyLanguages(tLang)){
        	throw new Exception(" Values of target language are not correct ");
        }

        Find_Best_Source temp = new Find_Best_Source();
        temp.getBestSourceLang(dataFile,sLangs,tLang);
	}
}
