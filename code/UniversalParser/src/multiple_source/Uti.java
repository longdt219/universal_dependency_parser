package multiple_source;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

public class Uti {
	
	public static final String ENGLISH = "english";
	public static final String PORTUGUESE = "portuguese";
	public static final String TURKISH = "turkish";
	public static final String SWEDISH = "swedish";
	public static final String ARABIC = "arabic";
	public static final String BASQUE = "basque";
	public static final String CATALAN = "catalan";
	public static final String HUNGARIAN = "hungarian";
	public static final String SPANISH = "spanish";
	public static final String CHINESE = "chinese";
	public static final String DUTCH = "dutch";
	public static final String DANISH = "danish";
	public static final String GREEK = "greek";
	public static final String GERMAN = "german";
	public static final String ITALIAN = "italian";
	public static final String CZECH = "czech";
	public static final String FINNISH = "finnish";
	public static final String FRENCH = "french";
	public static final String IRISH = "irish";
	
	public static ArrayList<String> read_training_path_file(String file_path) throws IOException{
		ArrayList<String> result = new ArrayList<String>();
		FileInputStream fis = new FileInputStream(file_path); 
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = ""; 
		while ((line = br.readLine())!=null ){
			result.add(line.trim());
		}
		br.close(); 
		fis.close();
		return result; 
	}
	public static String get_embed_file(String fileName, String language) throws IOException{
		FileInputStream fis = new FileInputStream(fileName); 
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = ""; 
		while ((line = br.readLine())!=null ){
			line = line.trim(); 
			if (line.contains(language)) return line;
		}
		br.close(); 
		fis.close();
		System.out.println(" ==========================================="); 
		System.out.println(" EXPECTING ERRORS !!!! CANT FIND EMBEDDING FOR LANGUAGE " + language);
		System.out.println(" ==========================================="); 
		return "";
	}
	public static String get_training_file(ArrayList<String> file_list, String language){
		String result = "";
		for (String file_name : file_list) {
			if (file_name.contains(language) && file_name.contains("train"))
				return file_name;
		}
		return ""; 
	}
	
	public static String get_testing_file(ArrayList<String> file_list, String language){
		String result = "";
		for (String file_name : file_list) {
			if (file_name.contains(language) && file_name.contains("test"))
				return file_name;
		}
		return ""; 
	}

	public static String get_dev_file(ArrayList<String> file_list, String language){
		String result = "";
		for (String file_name : file_list) {
			if (file_name.contains(language) && file_name.contains("dev"))
				return file_name;
		}
		return ""; 
	}

	public static boolean verifyLanguages(String lang_string){
		String[] langs = lang_string.split(",");
		
		for (String lang : langs){
			boolean check = false; 
			if (lang.equals(ARABIC)) check = true;  
			if (lang.equals(ENGLISH)) check = true; 
			if (lang.equals(PORTUGUESE)) check = true; 
			if (lang.equals(TURKISH)) check = true;
			if (lang.equals(SWEDISH)) check = true;
			if (lang.equals(BASQUE)) check = true;
			if (lang.equals(CATALAN)) check = true;
			if (lang.equals(HUNGARIAN)) check = true;
			if (lang.equals(SPANISH)) check = true;
			if (lang.equals(CHINESE)) check = true;
			if (lang.equals(DUTCH)) check = true;
			if (lang.equals(DANISH)) check = true;
			if (lang.equals(GREEK)) check = true;
			if (lang.equals(GERMAN)) check = true;
			if (lang.equals(ITALIAN)) check = true;
			if (lang.equals(CZECH)) check = true;
			if (lang.equals(FINNISH)) check = true;
			if (lang.equals(FRENCH)) check = true; 
			if (lang.equals(IRISH)) check = true; 
			if (!check) return false; 
		}
		return true;
	}
	
    /**
     * Returns the Jensen-Shannon divergence.
     */
    public static double jensenShannonDivergence(double[] p1, double[] p2) {
      assert(p1.length == p2.length);
      double[] average = new double[p1.length];
      for (int i = 0; i < p1.length; ++i) {
        average[i] += (p1[i] + p2[i])/2;
      }
      return (klDivergence(p1, average) + klDivergence(p2, average))/2;
    }

    
   public static final double log2 = Math.log(2);
    /**
     * Returns the KL divergence, K(p1 || p2).
     *
     * The log is w.r.t. base 2. <p>
     *
     * *Note*: If any value in <tt>p2</tt> is <tt>0.0</tt> then the KL-divergence
     * is <tt>infinite</tt>. Limin changes it to zero instead of infinite. 
     * 
     */
    public static double klDivergence(double[] p1, double[] p2) {


      double klDiv = 0.0;

      for (int i = 0; i < p1.length; ++i) {
        if (p1[i] == 0) { continue; }
        if (p2[i] == 0.0) { continue; } // Limin

      klDiv += p1[i] * Math.log( p1[i] / p2[i] );
      }

      return klDiv / log2; // moved this division out of the loop -DM
    }
	
	private static HashMap<String,ArrayList<Double>> readEmbeddingFile(String fileName) throws IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		HashMap<String,ArrayList<Double>> result = new HashMap<String, ArrayList<Double>>();
		String line = "";
		while ((line=br.readLine())!=null){
			String[] tokens = line.split("\\s+");
			ArrayList<Double> vector = new ArrayList<Double>();
			for (int i =1 ; i<tokens.length; i++){
				vector.add(Double.parseDouble(tokens[i]));
			}
			result.put(tokens[0], vector);
		}
		br.close();
		fis.close();
		return result;
	}
    public static void Join2EmbeddingFile(String embeddingFile,
			String secEmbedding, String newEmbeddingFile) throws IOException {
		// Read the first file 
		HashMap<String, ArrayList<Double>> f1 = readEmbeddingFile(embeddingFile);
		HashMap<String, ArrayList<Double>> f2 = readEmbeddingFile(secEmbedding);
		// Write to the new file 
		FileOutputStream fos = new FileOutputStream(newEmbeddingFile);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
		
		
		for (String key : f1.keySet()){
			bw.write(key );
			ArrayList<Double> value1 = f1.get(key);
			ArrayList<Double> value2 = new ArrayList<Double>();
			for (int i =0; i < value1.size(); i++) value2.add(0.0);
			
			
			if (f2.containsKey(key)){
				value2 = f2.get(key);
			}
			for (double v : value1)
				bw.write(" " + v);
			for (double v : value2)
				bw.write(" " + v);
			bw.write("\n");
		}
		
		for (String key : f2.keySet()){
			if (!f1.containsKey(key)){
				bw.write(key );
				ArrayList<Double> value2 = f2.get(key);
				ArrayList<Double> value1 = new ArrayList<Double>(value2.size());
				for (int i =0; i < value2.size(); i++) value1.add(0.0);
				
				for (double v : value1)
					bw.write(" " + v);
				for (double v : value2)
					bw.write(" " + v);
				bw.write("\n");
			}
		}
		bw.close();
		fos.close();
	}
	public static String get_bidict_file(String bidictFile, String lang) throws IOException {
		// TODO Auto-generated method stub
		FileInputStream fis = new FileInputStream(bidictFile);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = "";
		while ((line = br.readLine()) != null){
			if (line.contains(lang)){
				return line;
			}
		}
		br.close();
		fis.close();
		return "";
	}

}
