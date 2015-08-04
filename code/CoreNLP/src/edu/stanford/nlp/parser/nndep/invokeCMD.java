package edu.stanford.nlp.parser.nndep;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.BufferedWriter;

public class invokeCMD {

	public void runCommandNotWait(String command, boolean outFlag){

		try {
			//System.out.println(" Running " + command);
            Process p = Runtime.getRuntime().exec(new String[]{"/bin/bash","-c",command});
			//Process p = Runtime.getRuntime().exec(command);
			//outFlag = false; 
            if (outFlag){
            	// Print result 
                BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String line=null;
                while((line=input.readLine()) != null) {
                    System.out.println(line);
                }
            }            
        } catch(Exception e) {
            System.out.println(e.toString());
            e.printStackTrace();
        }
	}

	public void runSimpleCommand(String command, boolean outFlag){

		try {
			System.out.println(" Running " + command);
			ProcessBuilder builder = new ProcessBuilder(new String[]{"/bin/sh","-c",command});
			builder.redirectErrorStream(true);
			Process p = builder.start();
            //Process p = Runtime.getRuntime().exec(new String[]{"/bin/sh","-c",command});
			//Process p = Runtime.getRuntime().exec(command);
			//outFlag = false; 
            if (outFlag){
            	// Print result 
                BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String line=null;
                while((line=input.readLine()) != null) {
                    System.out.println(line);
                }
            }
            int exitVal = p.waitFor();
            System.out.println();
            System.out.println(" == > Exited with error code "+exitVal + " for : " + command);
            System.out.println();
            
        } catch(Exception e) {
            System.out.println(e.toString());
            e.printStackTrace();
        }
	}
	
	public void runSimpleCommand(String command, BufferedWriter bw){

		try {
			System.out.println(" Running " + command);
			ProcessBuilder builder = new ProcessBuilder(new String[]{"/bin/sh","-c",command});
			builder.redirectErrorStream(true);
			Process p = builder.start();
            //Process p = Runtime.getRuntime().exec(new String[]{"/bin/sh","-c",command});
			//Process p = Runtime.getRuntime().exec(command);
			//outFlag = false; 
            
            BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line=null;
            while((line=input.readLine()) != null) {
            		bw.write(line +"\n");
            }
            
            int exitVal = p.waitFor();
            System.out.println("Exited with error code "+exitVal + " for : " + command);
            
        } catch(Exception e) {
            System.out.println(e.toString());
            e.printStackTrace();
        }
	}

	public String outputCommand(String command, boolean outFlag){
		String result = "" ;
		try {
			//System.out.println(" Running " + command);
            Process p = Runtime.getRuntime().exec(new String[]{"/bin/sh","-c",command});
			//Process p = Runtime.getRuntime().exec(command);
			//outFlag = false; 
            if (outFlag){
            	// Print result 
                BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String line=null;
                while((line=input.readLine()) != null) {
                    System.out.println(line);
                    result += line;
                }
            }
            int exitVal = p.waitFor();
            System.out.println("Exited with error code "+exitVal + " for : " + command);
            
        } catch(Exception e) {
            System.out.println(e.toString());
            e.printStackTrace();
        }
        return result ;
	}

}
