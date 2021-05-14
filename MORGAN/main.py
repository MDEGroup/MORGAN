from models import retrieve_similar_class
import os
def main_ten_fold():

    for i in range(1, 10):
        train_context = './Datasets/D_1/C1.1/train' + str(i) + '.txt'
        test_context_path = "./Datasets/D_1/C1.1/test_" + str(i) + "/"

        gt_path= "./Datasets/D_1/C1.1//gt_" + str(i) + "/"
        result = 'results_ecore_test_'+str(i)+'.txt'

        for file in os.listdir(test_context_path):
            retrieve_similar_class(train_context, test_context_path+file,gt_path+file, result,n_classes=4,n_items=4,size=2,recType='struct')

# def main_single_eval():
#     i = '4'
#     train_context = './Datasets/D_1/C1.1/train' + str(i) + '.txt'
#     test_context_path = "./Datasets/D_1/C1.1/test_" + str(i) + "_2/"
#
#     gt_path = "./Datasets/D_1/C1.1//gt_" + str(i) + "_2/"
#     result = './Results/results_xmi_test_' + str(i) + '.txt'
#
#     for file in os.listdir(test_context_path):
#         retrieve_similar_class(train_context, test_context_path + file, gt_path + file, result,n_classes=4,n_items=4,size=2,recType='struct')


main_ten_fold()