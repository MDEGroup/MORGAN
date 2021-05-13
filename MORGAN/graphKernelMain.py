from dataset_utilities import *
from models import choose_kernel_model,retrieve_similar_class, success_rate






def main_single_eval():
    #ten_folder_classes()
    # #dataset_path = 'C:/Users/claudio/Desktop/ecore_new/'
    root_path = 'C:/Users/claudio/Desktop/Morgan_conf/C1.3_xmi/'
    # # # #
    # # # #
    # # # ### filter by size
    # # # for fold in os.listdir(dataset_path):
    # # #     for file in os.listdir(dataset_path + fold):
    # # #         select_fileby_size(dataset_path + fold + '/' + file, dataset_path + fold, dest_path + fold)
    # #
    # #
    # #
    out_path='./test_categories/gt_cluster/C1.3_xmi/'
    filter_path = './test_categories/gt_cluster/all_no_dup/'
    for fold in os.listdir(root_path):
        aggregate_cluster_files(root_path+fold+'/', out_path, fold+'.txt')
    filter_file(out_path, filter_path)



    # for i in range (1,11):
    #     cluster_path ='C:/Users/claudio/Desktop/ten_fold_ecore_structure/test'+str(i)+'/'
    #     filter_path = './test_categories/test_'+str(i)+'/'
    #     split_path = './split_files/test_'+str(i)+'/'
    #     out_gt_path = 'C:/Users/claudio/Desktop/test_classes/gt_'+str(i)+'/'
    #     out_test_path = 'C:/Users/claudio/Desktop/test_classes/test_'+str(i)+'/'
    #     for file in os.listdir(cluster_path):
    #         remove_duplicates(cluster_path+file,filter_path+file)
    #
    #     for file in os.listdir(filter_path):
    #         with open(filter_path+file, 'r') as f:
    #             num=int(len(f.readlines())/2)
    #             split_test_files(filter_path+file, num, file, out_gt_path, out_test_path)

        # for file in os.listdir(split_path):
        #     print(file)
        #     if str(file).find('_1') != -1:
        #         shutil.copy(split_path+file, out_test_path+file)
        #     else:
        #         shutil.copy(split_path + file, out_gt_path + file)



    #create_ten_fold_structure('C:/Users/claudio/Desktop/train_dataset_xmi/')
    # create_ten_fold_structure(dest_path)
    # for i in range (1,11):
    #
    #     fold_path ='./train_root/train_partial_'+str(i)+'/'
    #     out_path='./train_root/train_main/'
    #     filename ='train_partial_'+str(i)+'.txt'
    #     aggregate_cluster_files(fold_path, out_path, filename)



# def main_recommend_attributes():
#
#
#     #path_file= './train_no_duplicates.txt'
#
#     path_root='C:/Users/claudio/Desktop/ten_fold_gnn_ecores/test_files/'
#     tot_p =0
#     tot_r = 0
#     tot_f = 0
#     with open('results.csv', 'w', encoding='utf-8', errors='ignore') as out:
#         out.write('precision, recall, fmeasure \n')
#         for test in os.listdir(path_root):
#             G_train, G_test, labels, recommendations = get_graphs_cross_eval(path_root+test)
#
#
#             #compute_cross_validation(G_train, labels)
#             y_pred = choose_kernel_model(G_train,G_test,labels)
#             p, r, f = compute_metrics(y_pred, labels)
#
#             for l, rec, pred in zip(labels, recommendations, y_pred):
#                 if l == pred:
#                     print(rec)
#
#             out.write(str(p) + ',' + str(r) + ',' + str(f) + '\n')
#
#             tot_p = tot_p + p
#             tot_r = tot_r + r
#             tot_f = tot_f + f
#
#         print("avg precision ",tot_p / 10)
#         print("avg recall " ,tot_r / 10)
#         print("avg fmeasure ", tot_f / 10)


def main_recommend_attributes():
    #mapping_files()
    with open ('ecore_recommendations_C2.3.txt', 'w', encoding='utf-8', errors='ignore') as outcomes:


            for i in range (1, 11):
                print("-"*100)
                outcomes.write('round'+str(i)+'\n')
                print('round', i)

                dataset_path = 'C:/Users/claudio/Desktop/ten_fold_gnn_ecore_C1.1/train_'+str(i)+'/'
                out_path = './'
                filename = "train_ecores.txt"
                aggregate_cluster_files(dataset_path, out_path, filename)
                remove_duplicates('train_ecores.txt', 'train_no_duplicates.txt')
                path_train = './train_no_duplicates.txt'
                path_test = 'C:/Users/claudio/Desktop/ten_fold_gnn_ecore_C1.1/test_files/test'+str(i)+'.txt'
                G_train, G_test, y_train, y_test = get_original_graphs(path_train, path_test)

                y_pred = choose_kernel_model(G_train, G_test, y_train,model='weis',classfier='svm')

                for test, rec in zip(y_test, y_pred):
                    if test == rec:
                        print('match')
                    else:
                        print('no')




                # out.write(str(pr)+','+str(rec)+','+str(f)+ '\n')
                #
                # tot_p = tot_p + pr
                # tot_r = tot_r + rec
                # tot_f = tot_f + f

            #     for l, rec, pred in zip(y_test, recommendations, y_pred):
            #         if l == pred:
            #             outcomes.write(rec+'\n')
            # out.write(str(tot_p/10) + ',' + str(tot_r/10) + ',' + str(tot_f/10) + '\n')


def main_recommend_classes():

    for i in range(7,11):
        #train_context = "./test_categories/gt_cluster/gt_"+i+".txt"
        train_context = 'C:/Users/claudio/Desktop/ten_fold_gnn_xmi_C1.2/train_'+str(i)+'/train'+str(i)+'.txt'
        #remove_duplicates(train_context, 'train_no_duplicates.txt')
        test_context_path = "C:/Users/claudio/Desktop/Morgan_conf/C2.1_xmi/test_"+str(i)+"/"
        #test_context_path = 'C:/Users/claudio\Desktop\ten_fold_gnn_ecore_C1.1\test_files/

        gt_path= "C:/Users/claudio/Desktop/Morgan_conf/C2.1_xmi/gt_"+str(i)+"/"
        result = 'results_xmi_test_'+str(i)+'.txt'

        for file in os.listdir(test_context_path):
            retrieve_similar_class(train_context, test_context_path+file,gt_path+file, result)




#emf_compare_statistics("C:/Users/claudio/Desktop/output/webserver/com.google.oauth-client__google-oauth-client-jetty___1.31.4.xmi")
#main_recommend_attributes()
#path="C:/Users/claudio/Desktop/Morgan_conf/C1.1_ecore/gt_3/"

# for file in os.listdir(path):
#     list_gt = get_gt_class(path+file)
#     list_rec = get_rec_class('./results_classes_ecore_test_3.txt')
#     compute_edit_distance(list_gt,list_rec)
#main_recommend_classes()
#ten_folder_classes()
# path='./test_categories/test_4/'
# # for cluster in os.listdir(path):
# #     remove_duplicates(path+cluster, path+cluster.replace('.txt','_no_dup.txt'))
# compute_statistics(path+'webserver_no_dup.txt' )

#main_single_eval()

# root='C:/Users/claudio/Desktop/output/'
# for cat in os.listdir(root):
#     results=find_similar_model('C:/Users/claudio/Desktop/output/'+cat+'/')
#     move_similar_file("C:/Users/claudio/Desktop/filtered_xmi/"+cat+"/", "C:/Users/claudio/Desktop/similar_xmi/"+cat+"/", results)

#ten_folder_classes()
#success_rate('cosine_results_2.txt')

main_recommend_classes()

#main_recommend_attributes()

# src_path='C:/Users/claudio/Desktop/ten_fold_ecore_structure/test10/'
# filter_path = 'C:/Users/claudio/Desktop/raw_test_ecore/test10/'
# filter_file(src_path, filter_path)