package org.eclipse.ecore.parser;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.URI;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EClassifier;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;

public class EcoreParser {
    public static HashMap<String,HashMap<String,HashMap<String, String>>> getEcoreInfo(File ecoreFile) throws IOException{
    		
        ResourceSet resourceSet = new ResourceSetImpl();
        resourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put(
                "ecore", new EcoreResourceFactoryImpl());
        Resource myMetaModel = null;

        HashMap<String,HashMap<String,HashMap<String, String>>> ecoreElements = new HashMap<>();
        

        try {
            myMetaModel = resourceSet.getResource(URI.createFileURI(String.valueOf(ecoreFile)), true);
        }
        catch(Exception e){
            System.out.println(e);
            System.out.println(ecoreFile);
        }

        if(myMetaModel!=null) {
        	 EList<EObject> contents = myMetaModel.getContents();
            try {
            	 for (int i = 0; i < contents.size(); i += 1) {
               EPackage univEPackage = (EPackage) myMetaModel.getContents().get(i);
               
               ArrayList<EPackage> results = new ArrayList<EPackage>();
               ArrayList<EPackage> packages = new ArrayList<EPackage>();
               
               packages=exploreEcore(univEPackage, results);
               
               
               for (EPackage sub : packages) {

            	   for (EClassifier eClassifier : sub.getEClassifiers()) {

	                    if (eClassifier instanceof EClass) {
	                        EClass clazz = (EClass) eClassifier;
	                        
	                        if(clazz.getEAttributes().isEmpty() )  {
	                        	if (clazz.getEReferences().isEmpty()) {
	                        		continue;
	                        	}
	                        }                                                
	                       
	                        HashMap<String,HashMap<String, String>> categoryElements = new HashMap<>();                
	
	                        HashMap<String, String> attributes = new HashMap<String, String>();
	                        for (EAttribute eAttribute : clazz.getEAttributes()) {
	                            attributes.put(eAttribute.getName(), eAttribute.getEType().getName());                    

	                        }
	                        categoryElements.put("attributes",attributes);
	
	                        HashMap<String, String> reference = new HashMap<String, String>();
	                        for (EReference eReference : clazz.getEReferences()) {	                        	
	                            reference.put(eReference.getName(),eReference.getEType().getName());                         
	    
	                        }
	                        categoryElements.put("reference",reference);	                      
	                        ecoreElements.put(clazz.getName(),categoryElements);
                    }
                   
                }
               }
            	 }
            }
            catch (Exception e) {
                System.out.println(e);
                System.out.println(ecoreFile);
            }

        }    
      

        return ecoreElements;
    }
    
    public static void getEcoreData (HashMap<String,HashMap<String,HashMap<String, String>>> ecoreList, File filename,String label, FileWriter statWriter) {
    	
    	int coutClass = 0;
		int countAttr = 0;
		int countRel = 0;
    	try {
    		
    		
			FileWriter fw = new FileWriter(filename);
			
			StringBuilder featureBuilder = new StringBuilder();
			
			
			featureBuilder.append(filename.getName()+"," );
			
			for (String key : ecoreList.keySet()) {
				HashMap<String,HashMap<String,String>> elem = ecoreList.get(key);
				
				StringBuilder sb = new StringBuilder();				
				sb.append(label+"\t"+key + " ");				
				coutClass++;
				HashMap<String, String> attributeMap = elem.get("attributes");
				
				for(String attrName : attributeMap.keySet()) {					
					String typeAttr= attributeMap.get(attrName);
					countAttr++;
					sb.append('('+attrName+','+typeAttr+','+"attribute)"+" ");
				}
				
				HashMap<String, String> refMap = elem.get("reference");				
				
				for(String refName : refMap.keySet() ) {					
					String typeRef= refMap.get(refName);
					countRel++;
					sb.append('('+refName+','+typeRef+','+"ref)"+" ");
				}			
				
				
				fw.write(sb.toString()+"\n");
			}
			featureBuilder.append(coutClass+","+countAttr+","+countRel+",model\n");			
			statWriter.write(featureBuilder.toString());
			
			
			
			fw.flush();
			fw.close();
		} catch (IOException e) {			
			e.printStackTrace();
		}
    
    	
    	
    	
    }
    
    public static ArrayList<EPackage> exploreEcore(EPackage univEPackage, ArrayList<EPackage> results){
        EList<EPackage> subPackages = getSubPackage(univEPackage);

        if(results.size()<1&&subPackages.size()<1){results.add(univEPackage);}

        for(int i=0; i<subPackages.size(); i+=1){

            if(getSubPackage(subPackages.get(i)).size()>0){
                exploreEcore(subPackages.get(i), results);
            }
            else{                
                results.add(subPackages.get(i));
              
            }

        }
        return results;

    }
    
    public static EList<EPackage> getSubPackage(EPackage pack){
        return pack.getESubpackages();
    }
	
    
    

}
