package improve_supervised;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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

public class SupervisedMalt_Learning_Curve  implements Runnable{

	private String train_data;
	private String test_data; 
	private String Lang, Size;
	private String prefix; 
	
	public SupervisedMalt_Learning_Curve(String train, String test, String Lang, String Size, String prefix){
		this.train_data = train; 
		this.test_data = test; 
		this.Lang = Lang; 
		this.Size = Size;
		this.prefix = prefix;
	}
	public double get_measurement(String fileName) throws IOException{
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		String line = ""; 
		
		while ((line = br.readLine())!=null){
			line = line.trim(); 
			String[] tokens = line.split("\\s+");
			if (tokens.length ==3)
				if (tokens[1].equals("Row") && tokens[2].equals("mean")){
					return Double.parseDouble(tokens[0]);
				}
		}
		br.close();fis.close();
		return 0 ;
	}
	public void run() {
		// Run each supervised NN parser 
		int ID = (int) (1000000 * Math.random());
		String model_name = "model." + Lang +"." + ID;
		String out_file = this.prefix + "." + Lang +"." + Size;
		String out_temp = "malt.out." + ID;
		String las_file = "temp.las." + ID;
		String uas_file = "temp.uas." + ID; 
		// Model file : model.sv.dev
		invokeCMD ivk = new invokeCMD();
		
		// Train the model 
		String cmd = String.format("java -jar ../../tools/maltparser-1.8/maltparser-1.8.jar -c %s -i %s -m learn -a nivrestandard",model_name,this.train_data);
		ivk.runSimpleCommand(cmd, true);
		// Parse the test file 
		cmd = String.format("java -jar ../../tools/maltparser-1.8/maltparser-1.8.jar -c %s -i %s -m parse -o %s",model_name,this.test_data, out_temp);
		ivk.runSimpleCommand(cmd, true);
		// Check the result 
		cmd = String.format("java -jar ../../tools/MaltEval/lib/MaltEval.jar --Metric LAS -s %s -g %s --pattern '##.####' > %s", out_temp,this.test_data, las_file);
		ivk.runSimpleCommand(cmd, true);
		cmd = String.format("java -jar ../../tools/MaltEval/lib/MaltEval.jar --Metric UAS -s %s -g %s --pattern '##.####' > %s", out_temp,this.test_data, uas_file);
		ivk.runSimpleCommand(cmd, true);
		// Find and joint the result 
		try {
			double las = this.get_measurement(las_file);
			double uas = this.get_measurement(uas_file);
			// Write to out file 
			FileOutputStream fos = new FileOutputStream(out_file);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos,"UTF-8"));
			bw.write(uas+"\n");
			bw.write(las+"\n");
			bw.close(); fos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	

	public static void main(String[] args) throws Exception{
		
        CommandLineParser parser = new PosixParser();
        Options options = new Options();

        options.addOption(OptionBuilder.withLongOpt("Data").withDescription("Path to the file list all data path ").isRequired().hasArg().withArgName("data").create("data"));        
        options.addOption(OptionBuilder.withLongOpt("Langs").withDescription("Set of languages ").isRequired().hasArg().withArgName("Langs").create("Langs"));
        options.addOption(OptionBuilder.withLongOpt("DataPoints").withDescription("Data point (in k) for running ").isRequired().hasArg().withArgName("range").create("range"));
        options.addOption(OptionBuilder.withLongOpt("Thread").withDescription("Number of Threads").isRequired().hasArg().withArgName("thread").create("thread"));
        options.addOption(OptionBuilder.withLongOpt("Prefix").withDescription("Ouput Prefix (e.g. result.mst.sup) ").hasArg().withArgName("prefix").create("prefix"));
        
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
        String dataPoints = commandLine.getOptionValue("DataPoints");
        String prefix = "result.malt.sup";
        if (commandLine.hasOption("Prefix"))
        	prefix = commandLine.getOptionValue("Prefix");
        	
        int thread_no = Integer.parseInt(commandLine.getOptionValue("Thread"));
        
        if (!Uti.verifyLanguages(Langs)){
        	throw new Exception(" Values of source languages are not correct ");
        }
        
        ExecutorService es = Executors.newFixedThreadPool(thread_no);
        
        String[] dataList = dataPoints.split(",");
        String[] langList = Langs.split(",");
        ArrayList<String> file_list = Uti.read_training_path_file(dataFile);
        for (String data_size : dataList){
        	// 2. Run the supervised with all the languages (each supervised => one thread)
        	for (String lang : langList){
        		String test_file = Uti.get_testing_file(file_list, lang);
        		String train_file = Uti.get_training_file(file_list, lang);
        		String small_train_file = train_file + "." + data_size;
                SupervisedMalt_Learning_Curve temp = new SupervisedMalt_Learning_Curve(small_train_file, test_file, lang, data_size , prefix);
                es.execute(temp);
        	}

        }
		es.shutdown();
		es.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);		
	}

}
