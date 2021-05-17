package it.disim.univaq.modisco.injector.handlers;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

public class SampleHandler extends AbstractHandler {

//	private final String projectName = "unamur.pol";
//	private final String rootSourcesPath = "/Users/juri/Desktop/test/";
//	private boolean export = true;
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		ModelInjector testissimo = new ModelInjector();
		try {
//			testissimo.getModelFormJar("/Users/juri/Desktop/org.eclipse.modisco.infra.discovery.benchmark.core_1.5.0.v20210214-0825.jar");
			testissimo.computeAllModelsFromAFolder("/Users/juri/Desktop/testJar", "/Users/juri/Desktop/testJarO");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
		
		//BUONO
//		try {
//			JavaProjectFactory jpf = new JavaProjectFactory(this.projectName);
//			IJavaProject javaProject = jpf.getJavaProject();
//			
//
//			String sourceFolderPath = this.rootSourcesPath + this.projectName; //$NON-NLS-1$
//			URL src = Activator.getDefault().getBundle().getEntry(sourceFolderPath);
//			jpf.populateSourceFolder(sourceFolderPath,
//					Activator.getDefault());
//
//			DiscoverJavaModelFromJavaProject javaDiscoverer = new DiscoverJavaModelFromJavaProject();
//			javaDiscoverer.discoverElement(javaProject, new NullProgressMonitor());
//			Resource output = javaDiscoverer.getTargetModel();
//
//			if (this.export) {
//				File f = new File(projectName);
//				FileOutputStream fout = new FileOutputStream(f);
//				output.save(fout, Collections.EMPTY_MAP);
//				fout.close();
//			}
//		} catch (CoreException | DiscoveryException | IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		return null;
	}
}
