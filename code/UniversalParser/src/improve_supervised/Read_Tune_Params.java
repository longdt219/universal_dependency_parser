package improve_supervised;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import multiple_source.Uti;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

import multiple_source.*;

public class Read_Tune_Params {
	
	public static String read_result(String fileName, String flag) {
		try{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String uas = br.readLine().trim();
		String las = br.readLine().trim();
		br.close();
		fis.close();

		if (flag.equals("UAS")) return uas; 
		else return las;
		}
		catch (Exception e){
			return "-10000";
		}
	}
	

	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

                
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("DataPoints").withDescription("Data point (in k) for running ").isRequired().hasArg().withArgName("range").create("range"));
        options.addOption(OptionBuilder.withLongOpt("Regularlization").withDescription("Regularlization settings ").isRequired().hasArg().withArgName("reg").create("reg"));
        options.addOption(OptionBuilder.withLongOpt("Prefix").withDescription("Which prefix is readed followed by (.lang.size.reg)").hasArg().withArgName("prefix").create("prefix"));
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


        String Langs = commandLine.getOptionValue("Langs");
        String dataPoints = commandLine.getOptionValue("DataPoints");
        String regs = commandLine.getOptionValue("Regularlization");
        
        String prefix = "result.tune"; 
        if (commandLine.hasOption("Prefix"))
        	prefix = commandLine.getOptionValue("Prefix");
        
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        
        String[] dataList = dataPoints.split(",");
        String[] langList = Langs.split(",");
        String[] regList = regs.split(",");
        
        System.out.println(" UAS :");
        for (String reg : regList) {
        	System.out.print(reg +",");
        	for (String data_size : dataList){
        		double avg = 0; 
            	for (String lang : langList){
            		String result_file = prefix + "." + lang + "." + data_size + "." + reg;
            		avg += Double.parseDouble(read_result(result_file, "UAS")); 
            	}
            	System.out.print(avg/ langList.length +",");        		
        	}
        	System.out.println();
        }
        
        System.out.println(" LAS :");
        
        for (String reg : regList) {
        	System.out.print(reg +",");
        	for (String data_size : dataList){
        		double avg = 0; 
            	for (String lang : langList){
            		String result_file = prefix + "." + lang + "." + data_size + "." + reg;
            		avg += Double.parseDouble(read_result(result_file, "LAS")); 
            	}
            	System.out.print(avg/ langList.length +",");        		
        	}
        	System.out.println();
        }

	}

}
