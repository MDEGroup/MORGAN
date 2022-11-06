from GNN_engine import get_recommendations
import os, time
from dataset_utilities import enrich_data, mapping_vce_terms
import nltk

def main():


    for i in range(1, 11):

        print('Start round ', i)

        root ="C:\\Users\\claud\\OneDrive\\Desktop\\XES_ten_folder_small\\"

        train_context = root+'train' + str(i) + '.txt'
        test_context = root+'test'+str(i)+'/'
        gt_path = root+'gt_'+str(i)+'/'
        result = root+'/results_round'+str(i)+'.csv'

        preprocessed_train, train_data, labels = enrich_data(train_context)

        for file in os.listdir(test_context):
            get_recommendations(train_preprocessed=preprocessed_train,train_data=train_data, test_context=test_context+file,
                                result_file=result, n_classes=10, n_items=10, size=2, recType='class')



if __name__ == "__main__":
    main()