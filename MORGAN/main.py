from GNN_engine import get_recommendations
import os


for i in range(1, 10):
    train_context = './Datasets/D_beta/C1.1/train' + str(i) + '.txt'
    test_context = './Datasets/D_beta/C1.1/test_' + str(i) + '/'

    gt_path= './Datasets/D_beta/C1.1/gt_' + str(i) + '/'
    result = './Results/results_ecore_test_'+str(i)+'.txt'

    for file in os.listdir(test_context):
        get_recommendations(train_context=train_context, test_context=test_context+file,gt_context=gt_path+file,
                            result_file=result,n_classes=4,n_items=4,size=2,recType='struct')

