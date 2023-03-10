import sys

from GNN_engine import get_recommendations

import os, time
from dataset_utilities import enrich_data
# from ranker import calculate_clusters
# from matplotlib import pyplot as plt
# from dataset_utilities import plot_dendrogram, read_similarity_matrix
# import pandas as pd

#import nltk


# def compute_similarities():
#     models_path = "C:/Users/claud/OneDrive/Desktop/Ecore_models/"
#     out_path = "Ecore_matrix.csv"
#     #
#     similarity_matrix = compute_semantic_similarity(models_path,out_path)
#     #similarity_matrix = read_similarity_matrix("CDM_matrix.csv")
#     clusters = calculate_clusters(similarity_matrix)
#     plt.title("Ecore Dendrogram")
#     # plot the top three levels of the dendrogram
#     plot_dendrogram(clusters)
#     plt.xlabel("Ecore models")
#     plt.show()
#     #plt.savefig("cdm_dendogram.pdf", format="pdf")


def main(data_path, n_classes, n_items, size, rec_type):


    for i in range(1, 11):
        print('Start round ', i)

        #data_path = "C:/Users/claud/OneDrive/Desktop/Grakel/Grakel/Datasets/D_beta/C2.1/"

        train_context = data_path+'train' + str(i) + '.txt'
        test_context = data_path+'test_'+str(i)+'/'
        gt_context = data_path+'gt_'+str(i)+'/'
        result_file = data_path+'/results_round'+str(i)+'.csv'

        preprocessed_train, train_data, labels = enrich_data(train_context)




        for file in os.listdir(test_context):
            get_recommendations(train_preprocessed=preprocessed_train,train_data=train_data, test_context=gt_context+file,
                                result_file=result_file, n_classes=n_classes, n_items=n_items, size=size, rec_type=rec_type)



if __name__ == "__main__":
    #compute_similarities()
    main(data_path=sys.argv[1], n_classes=int(sys.argv[2]), n_items=int(sys.argv[3]), size=int(sys.argv[4]),rec_type=sys.argv[5])