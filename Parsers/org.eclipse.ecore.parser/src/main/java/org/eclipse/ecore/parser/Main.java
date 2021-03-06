package org.eclipse.ecore.parser;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;


public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub		
		ResourceSet resourceSet = new ResourceSetImpl();
        resourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put(
                "ecore", new EcoreResourceFactoryImpl());
        String srcPath = "input_ecores\\";
        String destPath = "out_ecores\\";
        File rootDir = new File(srcPath);
        System.out.println("Exctracting features from ecore files...");
        for (File fold: rootDir.listFiles()) {
        	File dir = new File(fold+"\\");
        	File[] directoryListing = dir.listFiles();
        	for (File f : directoryListing) {
    	        try {
    	        	
    	        	HashMap<String,HashMap<String,HashMap<String, String>>> list=EcoreParser.getEcoreInfo(f);    	        	
    	        	File outfile = new File(destPath+fold.getName()+"\\"+f.getName()+"_results.txt");
    	        	EcoreParser.getEcoreData(list, outfile, fold.getName());
    			} catch (IOException e) {
    				
    				e.printStackTrace();
    			}   	        
    	        
    	        
    		}
        }
        System.out.println("Done.");
  
	}

}
