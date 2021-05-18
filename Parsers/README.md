## MORGAN parsers

This subfolder contains the parsers used to extract the data from metamodels and models. They are structured as follows

 - **org.eclipse.ecore.parsers** project extracts the metaclasses and the corresponding structural features using the EMF utilities. The original ecore files are stored in the *input_ecore* folder.
 - **org.eclipse.modisco.parser** project contains the MoDisco utilities employed to extract classes and the corresponding class members from the xmi files stored in the *input_xmi* folder

To extract the data from metamodels and models, you have to run the **Main.java** file for each project. The results will be stored in the **out_ecore** and **out_xmi** according to the type of parser.
