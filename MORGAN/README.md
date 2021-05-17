##  MORGAN implementation

This subfolder contains the Python implementation of MORGAN as well as the datasets used in the evaluation.
## Environment setup
To run the tool, you need to install the following python libraries:

 - grakel 0.1.8
 - networkx >2.5 
 - nltk >3.5

## Folder structure 

The folder is structured as follows:
```
 	|--- Datasets		It contains the training, testing, and ground truth files 											
				for all datasets and configurations
				
	|--- Results 		This folder stores the results of the metrics computation 			   
				process
						
	|--- utilities.py   	It contains functions used to preprocess the textual    								
	                    	textual files
				
	|--- GNN_engine.py	It contains functions used to enable the GNN engine
	
	|--- main.py		It performs the ten-fold validation 
      
```

## Running example

To replicate the experiment, you need to specify the following parameters in the **main.py** file:

 - *train_context*: path to the train files 
 - *test_context:* path to test files
 - *gt_context:* path to the ground truth files
 - *result:* path to store the results 
 - *n_classes*: number of recommended metaclasses/classes 
 - *n_items:*: number of recommended structural features/class members
 - *size:* size of the testing models (=1 for Π/3, =2 for 2Π/3)
 - *recType:* a string value to specify the type of retrieved recommendation (='class' for the metaclasses/classes, ='struct' for structural features/class members)


A possible configuration the **get_recommendations** function for one round can be the following:

    get_recommendations(train_context= './Datasets/D_beta/C1.1/train1.txt', test_context='./Datasets/D_beta/C1.1/test_1/'+filename,gt_context= './Datasets/D_beta/C1.1/gt_1/',  result_file='./Results/results_ecore_test_1.txt',n_classes=4,n_items=4,size=2,recType='struct')
In particular, the above setting produces the results for metamodel's structural features using the configuration C1.1 
