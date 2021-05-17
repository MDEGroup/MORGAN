package it.disim.univaq.modisco.injector.handlers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.core.runtime.Path;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.jdt.core.IClasspathEntry;
import org.eclipse.jdt.core.IJavaProject;
import org.eclipse.jdt.core.IPackageFragmentRoot;
import org.eclipse.jdt.core.JavaCore;
//import org.eclipse.jdt.core.dom.BodyDeclaration;
import org.eclipse.jdt.core.dom.ImportDeclaration;
import org.eclipse.jdt.launching.IVMInstall;
import org.eclipse.jdt.launching.JavaRuntime;
import org.eclipse.jdt.launching.LibraryLocation;
import org.eclipse.modisco.java.AbstractTypeDeclaration;
import org.eclipse.modisco.java.Archive;
import org.eclipse.modisco.java.ClassFile;
import org.eclipse.modisco.java.FieldDeclaration;
import org.eclipse.modisco.java.MethodDeclaration;
import org.eclipse.modisco.java.Model;
import org.eclipse.modisco.java.Package;
import org.eclipse.modisco.java.Type;
import org.eclipse.modisco.java.TypeAccess;
import org.eclipse.modisco.java.VariableDeclarationFragment;
import org.eclipse.modisco.java.discoverer.DiscoverJavaModelFromLibrary;
import org.eclipse.modisco.java.emf.JavaPackage;
import org.eclipse.modisco.java.BodyDeclaration;
import org.eclipse.modisco.java.ClassDeclaration;

public class ModelInjector {

	private final String fakeProject = "FakeProject";

	public void computeAllModelsFromAFolder(String inputFolder, String outputFolder) {
		File dir = new File(inputFolder);
		File[] directoryListing = dir.listFiles();
		if (directoryListing != null)
			for (File child : directoryListing)
				if (child.getName().endsWith(".jar"))
					getModeFormJarlAndSerialize(child.getAbsolutePath(),
							Paths.get(outputFolder, child.getName().replace(".jar", ".xmi")).toString());
	}

	public Resource getModelFormJar(String jar) throws Exception {

		if (jar == null)
			throw new Exception("jar String is null");

		/* Open eclipse project */
		JavaProjectFactory jpf = new JavaProjectFactory(fakeProject);
		IProject project = jpf.getPoject();
		IJavaProject javaProject = jpf.getJavaProject();
		IProgressMonitor monitor = new NullProgressMonitor();
		IPath location = new Path(jar);
		IFile file = project.getFile(location.lastSegment());
		file.createLink(location, IResource.REPLACE, monitor);
		IPath filePath = project.getWorkspace().getRoot().getLocation().append(file.getFullPath());

		/* Set project fragment roots */
		Set<IClasspathEntry> e = new HashSet<IClasspathEntry>();
		e.addAll(Arrays.asList(javaProject.getRawClasspath()));
		boolean entryExist = false;
		for (IClasspathEntry c : javaProject.getRawClasspath()) {
			if (filePath.toOSString().matches(c.getPath().toOSString())) {
				entryExist = true;
				break;
			}
		}
		IVMInstall vmInstall = JavaRuntime.getDefaultVMInstall();
		LibraryLocation[] locations = JavaRuntime.getLibraryLocations(vmInstall);
		for (LibraryLocation element : locations) {
			e.add(JavaCore.newLibraryEntry(element.getSystemLibraryPath(), null, null));
		}
		for (IPackageFragmentRoot r : javaProject.getPackageFragmentRoots()) {
			if (filePath.toOSString().matches(r.getPath().toOSString())) {
				entryExist = true;
				break;
			}
		}
		if (!entryExist) {
			e.add(JavaCore.newLibraryEntry(filePath, null, null));
		}
		javaProject.setRawClasspath(e.toArray(new IClasspathEntry[e.size()]), monitor);

		/* Create discover for jar file */
		DiscoverJavaModelFromLibrary discover = new DiscoverJavaModelFromLibrary();
		IPackageFragmentRoot root = JavaCore.createJarPackageFragmentRootFrom(file);
		if (root == null) {
			throw new Exception("Error: Fragment root is null. Aborting...");
		}

		/* Create model from jar file */
		discover.discoverElement(root, monitor);
		jpf.deleteProhject();
		return discover.getTargetModel();

	}

	public void serializeResource(Resource output, String path) {
		File f = new File(path);
		FileOutputStream fout;
		try {
			fout = new FileOutputStream(f);
			output.save(fout, Collections.EMPTY_MAP);
			fout.close();
		} catch (FileNotFoundException e) {
		} catch (IOException e) {
		}
	}

	public void getModeFormJarlAndSerialize(String jar, String path) {
		try {
			serializeResource(getModelFormJar(jar), path);

		} catch (Exception e) {
			System.err.println(jar);
		}
	}

	public HashMap<String,HashMap<String,HashMap<String, String>>>  loadJavaModel(String projectPath,  String tag) {
		Model javaModel = null;
		HashMap<String,HashMap<String,HashMap<String, String>>> ecoreElements = new HashMap<>();
		try {
			EPackage.Registry.INSTANCE.put(JavaPackage.eNS_URI, JavaPackage.eINSTANCE);
			// register the default resource factory
			Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

			// create a resource set.
			ResourceSet resourceSet = new ResourceSetImpl();

			// get the URI of the model file.
			URI fileURI = URI.createFileURI(new File(projectPath).getAbsolutePath());

			// load the resource for this file
			Resource resourceModel = resourceSet.getResource(fileURI, true);
			

			// get the root element of the javamodel's project
			javaModel = (Model) resourceModel.getContents().get(0);

			// System.out.println(javaModel.getOrphanTypes());
			
			
			for (Archive b : javaModel.getArchives()) {
				List<ClassFile> listClass = b.getClassFiles();
				for (ClassFile c : listClass) {
					//System.out.println(c.getName());
					
					List<AbstractTypeDeclaration> ownedList = c.getPackage().getOwnedElements();					   
						
						for (AbstractTypeDeclaration elem :  c.getPackage().getOwnedElements()) {							
							
							List<BodyDeclaration> declarationList = elem.getBodyDeclarations();
							
							
							
								for (BodyDeclaration bd : declarationList) {									
									if (bd instanceof ClassDeclaration) {
										
										
										
										List<BodyDeclaration> bodyClass= ((ClassDeclaration) bd).getBodyDeclarations();
										HashMap<String,HashMap<String, String>> categoryElements = new HashMap<>();
										HashMap<String, String> methods = new HashMap<String, String>();
										HashMap<String, String> fields = new HashMap<String, String>();
										for (BodyDeclaration own: bodyClass) {
											if (own instanceof FieldDeclaration) {
												FieldDeclaration field = (FieldDeclaration) own;												
												fields.put(field.getFragments().get(0).getName(),field.getType().getType().getName());
												
												
											}
											
											if (own instanceof MethodDeclaration) {
												MethodDeclaration md = (MethodDeclaration) own;
												methods.put(md.getName(), md.getReturnType().getType().getName());
											}
											 
										}
										categoryElements.put("field", fields);
										categoryElements.put("method", methods);
										ecoreElements.put(bd.getName(),categoryElements);
										
										
									}
									
									
								
								
								}
							
							
						}

					
					
				}

			}
			

		} catch (Exception e) {
			System.out.println("An error occurred. The model can not be loaded.");
			
		}
		return ecoreElements;

	}
	
    public void getEcoreData (HashMap<String,HashMap<String,HashMap<String, String>>> ecoreList, File filename,String label) {
    	try {
			FileWriter fw = new FileWriter(filename);
			
			for (String key : ecoreList.keySet()) {
				HashMap<String,HashMap<String, String>> elem = ecoreList.get(key);				
				StringBuilder sb = new StringBuilder();
				
				
				sb.append(label+"\t"+key + " ");			
				
				HashMap<String, String> mdMap= elem.get("method");
				HashMap<String, String> fieldMap= elem.get("field");
				
				for(String mdName: mdMap.keySet()) {					
					String mdType = mdMap.get(mdName);
					sb.append('('+mdName+','+mdType+','+"method)"+" ");
				}
				
				for(String fieldName: fieldMap.keySet()) {					
					String fieldType = fieldMap.get(fieldName);
					sb.append('('+fieldName+','+fieldType+','+"field)"+" ");
				}				
				
				fw.write(sb.toString()+"\n");
			}
			
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	
    	
    }

}
