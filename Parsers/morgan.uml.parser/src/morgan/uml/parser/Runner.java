package morgan.uml.parser;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.FeatureNotFoundException;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceImpl;
import org.eclipse.jface.widgets.Property;
import org.eclipse.uml2.uml.Element;
import org.eclipse.uml2.uml.Model;
import org.eclipse.uml2.uml.PackageableElement;
import org.eclipse.uml2.uml.UMLPackage;
import org.eclipse.uml2.uml.internal.resource.UMLResourceFactoryImpl;

public class Runner {

	private static void parse(String inPath, String outPath) throws IOException, FeatureNotFoundException {

		File dir = new File(inPath);
		
		
		
		//int count = 0;
		
		//int countPackage = 0;
		
		File features = new File (outPath+"\\results.csv");	
		
		FileWriter featureWriter = new FileWriter(features);
		
		String header = "Name,MC,Attr,Ref,LABEL\n";
		
		
		
		
		

		for (File f : dir.listFiles()) {
			
			StringBuilder statBuilder = new StringBuilder();
			
			int coutClass = 0;
			int countAttr = 0;
			int countRel = 0;
			
			File outfile = new File(outPath+"\\"+f.getName()+"_results.txt");
			
			

			//URI uri = URI.createFileURI("input_uml//" + "___-M0L8xEeeEXb8Dudo6PQ.xmi");
			URI uri = URI.createFileURI(f.getAbsolutePath());
			System.out.println(f.getAbsolutePath());
			Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("ecore", new EcoreResourceFactoryImpl());
			Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("uml", new UMLResourceFactoryImpl());
			Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

			EPackage.Registry.INSTANCE.put(UMLPackage.eNS_URI, UMLPackage.eINSTANCE);

			ResourceSet rs = new ResourceSetImpl();
			rs.getPackageRegistry().put(UMLPackage.eNS_URI, UMLPackage.eINSTANCE);
			rs.getResourceFactoryRegistry().getExtensionToFactoryMap().put(".xmi", new XMIResourceImpl());
			
									
			try {
				FileWriter fw = new FileWriter(outfile);
				Resource resource=rs.getResource(uri, true);
				resource.load(null);
				Model content = (Model) resource.getContents().get(0);
				
				statBuilder.append(f.getName()+',');
				
				for (PackageableElement c : content.getPackagedElements()) {
					StringBuilder sb = new StringBuilder();
					if (c instanceof org.eclipse.uml2.uml.Class)						
						System.out.println("class " + c.getName());
						sb.append("model"+"\t"+ c.getName() + " ");
						coutClass++;
						
					
					for (Element a : c.getOwnedElements()) {
						
						if (a instanceof org.eclipse.uml2.uml.Property) {
							String attr = ((org.eclipse.uml2.uml.Property) a).getName().toString();					
							
							sb.append(attr+ ","+ "property ");
							countAttr++;
							System.out.println("attribute " + attr+ " ");
						}
						if (a instanceof org.eclipse.uml2.uml.Operation) {
							String operation = ((org.eclipse.uml2.uml.Operation) a).getName().toString();
							System.out.println("operation " + operation);
							countRel++;
							sb.append(operation+",operation ");
						}					
						

					}
					fw.write(sb.toString()+"\n");							
					
					
				}
				statBuilder.append(coutClass+","+countAttr+","+countRel+","+"model\n");
				featureWriter.write(statBuilder.toString());
				fw.flush();
				fw.close();
				
				
				
				
				// System.out.println(content);
			} catch (Exception e) {
				System.out.println(e.getMessage());
				
			}
			
			
		}
		
		featureWriter.flush();
		featureWriter.close();
		
		
		

//		ResourceSet set = new ResourceSetImpl();
//		
//		set.getPackageRegistry().put(UMLPackage.eNS_URI, UMLPackage.eINSTANCE);
//		set.getResourceFactoryRegistry().getExtensionToFactoryMap()
//		   .put(".xmi", new UMLResourceImpl(null));
//		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap()
//		   .put(".xmi", new UMLResourceImpl(null));
//		
//		Resource res = set.getResource(URI.createFileURI("/Users/juri/Downloads/modelset/raw-data/repo-genmymodel-uml/data/___-M0L8xEeeEXb8Dudo6PQ.xmi"), true);
//		res.getContents().get(0); // For example to get the root of your model.

	}

	public static void main(String[] args) throws IOException, FeatureNotFoundException {
		parse("input_uml//","out_uml//" );
	}
}
