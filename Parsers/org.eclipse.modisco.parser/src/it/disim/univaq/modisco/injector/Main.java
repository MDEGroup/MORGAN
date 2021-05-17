package it.disim.univaq.modisco.injector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import it.disim.univaq.modisco.injector.handlers.ModelInjector;

public class Main {
	public static void main (String[] args) {
		ModelInjector mi = new ModelInjector();
		
		String srcPath ="input_xmi\\";
		String dstPath= "out_xmi\\";
		File rootDir = new File (srcPath);
		System.out.println("Extracting features from models...");
		for (File fold : rootDir.listFiles()) {
					
			File dir = new File(fold+"\\");			
			File[] directoryListing = dir.listFiles();
			
			for (File f: directoryListing) {				
				HashMap<String,HashMap<String,HashMap<String, String>>> results = mi.loadJavaModel(f.getPath(),  fold.getName());		
				File outfile = new File(dstPath+fold.getName()+"\\"+f.getName()+"_result.txt");    
				mi.getEcoreData(results, outfile ,fold.getName());
		        
			}
		}
		
		System.out.println("Done.");

	}

}
