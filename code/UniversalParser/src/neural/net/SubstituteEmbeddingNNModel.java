package neural.net;

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
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class SubstituteEmbeddingNNModel {
	
	public static ArrayList<String> readEmbedFile(String fileName) throws IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		ArrayList<String> result = new ArrayList<String>();
		String line = ""; 
		while ((line = br.readLine())!=null){
			line = line.trim();
			result.add(line);
		}
		
		br.close(); fis.close();
		return result; 
	}
	
	public static String  readModelFile (String fileName, ArrayList<String> wordEmbed, ArrayList<String> posEmbed, ArrayList<String> theOther) throws IOException {
		String otherCommand = "";
		
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));	
		String line = "";
		int count = 0;
		int dictSize = 0; 
		int posSize = 0; 
		while ((line = br.readLine())!=null){
			line = line.trim();
			count ++; 
			if (count == 1)		dictSize = Integer.parseInt(line.split("=")[1]);
			if (count == 2) 	posSize = Integer.parseInt(line.split("=")[1]); 
			if ((count >= 3) && (count <=7)) otherCommand += line + "\n"; 
			if ((count >7) && (count <= dictSize+7))
				wordEmbed.add(line);
			if ((count > dictSize + 7) && (count <= posSize + dictSize + 7))
				posEmbed.add(line);
			if (count > posSize + dictSize + 7)
				theOther.add(line);
		}
		
		br.close(); fis.close();	
		return otherCommand;
	}
	
	public static void main(String[] args) throws IOException{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Model").withDescription("Path to the NN model ").isRequired().hasArg().withArgName("model").create("m"));        
        options.addOption(OptionBuilder.withLongOpt("WordEmbed").withDescription("Path to the Word Embedding").isRequired().hasArg().withArgName("word_embed").create("we"));
        options.addOption(OptionBuilder.withLongOpt("PosEmbed").withDescription("Path to the POS Embedding").hasArg().withArgName("pos_embed").create("pe"));
        options.addOption(OptionBuilder.withLongOpt("Output").withDescription("Output model file name").isRequired().hasArg().withArgName("output").create("o"));
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

        String modelFile = commandLine.getOptionValue("Model");
        String wEmbedFile = commandLine.getOptionValue("WordEmbed");
        String pEmbedFile = "";
        if (commandLine.hasOption("PosEmbed"))
        	pEmbedFile = commandLine.getOptionValue("PosEmbed");
        
        String outputFile = commandLine.getOptionValue("Output");
		
        ArrayList<String> we = SubstituteEmbeddingNNModel.readEmbedFile(wEmbedFile);
        
        ArrayList<String> pe = new ArrayList<String>(); 
        if (!pEmbedFile.equals(""))
        	pe = SubstituteEmbeddingNNModel.readEmbedFile(pEmbedFile);
        
		ArrayList<String> goldWe = new ArrayList<String>();
		ArrayList<String> goldPe = new ArrayList<String>();
		ArrayList<String> goldOther = new ArrayList<String>();
		String otherCmd = SubstituteEmbeddingNNModel.readModelFile(modelFile, goldWe, goldPe, goldOther);
		// Write to file
		FileOutputStream fos = new FileOutputStream(outputFile);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
		bw.write("dict="+(we.size() + 3) + "\n");
		if (!pEmbedFile.equals(""))
				bw.write("pos="+(pe.size() + 3) + "\n");
		else
				bw.write("pos="+goldPe.size() + "\n");
		bw.write(otherCmd);
		
		// Word Embedding 
		for (int i =0; i<3; i++) bw.write(goldWe.get(i) + "\n");
		for (int i =0; i<we.size(); i++) bw.write(we.get(i) + "\n");
		
		// Pos Embedding
		if (!pEmbedFile.equals("")){
			for (int i =0; i<3; i++) bw.write(goldPe.get(i) + "\n");
			for (int i =0; i<pe.size(); i++) bw.write(pe.get(i) + "\n");			
		}
		else {
			for (int i =0; i<goldPe.size(); i++) bw.write(goldPe.get(i) + "\n");
		}
		// The others 
		for (int i =0; i<goldOther.size(); i++) bw.write(goldOther.get(i) + "\n");
		
		bw.close();
		fos.close();
		
	}
}
