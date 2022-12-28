package morgan.uml.parser;

import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceImpl;
import org.eclipse.uml2.uml.UMLPackage;
import org.eclipse.uml2.uml.internal.resource.UMLResourceFactoryImpl;

public class Runner {

	private static void parse() throws IOException {
		URI uri = URI.createFileURI("/Users/juri/Downloads/modelset" + "/raw-data/repo-genmymodel-uml/data/"
				+ "___-M0L8xEeeEXb8Dudo6PQ.xmi");
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("ecore", new EcoreResourceFactoryImpl());
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("uml", new UMLResourceFactoryImpl());
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());
		
		EPackage.Registry.INSTANCE.put(UMLPackage.eNS_URI, UMLPackage.eINSTANCE);

		ResourceSet rs = new ResourceSetImpl();
		rs.getPackageRegistry().put(UMLPackage.eNS_URI, UMLPackage.eINSTANCE);
		rs.getResourceFactoryRegistry().getExtensionToFactoryMap().put(".xmi",
				new XMIResourceImpl());

		Resource resource = rs.getResource(uri, true);
		try {
			resource.load(null);
			EObject content = resource.getContents().get(0);
			System.out.println(content);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

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

	public static void main(String[] args) throws IOException {
		parse();
	}
}
