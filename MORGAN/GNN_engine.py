import os
import time

from grakel.kernels import  WeisfeilerLehmanOptimalAssignment
from grakel.utils import graph_from_networkx

from custom_kernel_matrix import CustomKernelMatrix
from dataset_utilities import get_vocab, create_graphs_of_words, precision, recall,\
    success_rate,  get_gt_classes,enrich_data, load_file, compute_graph_metrics, present_recommendations




def compute_recommendations(G_train, train_data, G_test, n, size):
    ranked_list = ()

    list_sim = []
    tot = 0
    for g, rec in zip(G_train, train_data):
        try:
            if len(G_test[n]) > 0:
                if size == 1:
                    tot = 3
                elif size == 2:
                    tot = (len(G_test[n]) * size) / 3

                if tot > 0:
                    sim = compute_kernel_similarity(g, G_test[n])
                    if sim[0][0] > 0:
                        tuple_g = rec, sim[0][0]
                        list_sim.append(tuple_g)
                    ranked_list = sorted(list_sim, key=lambda tup: tup[1], reverse=True)
            else:
                continue
        except IndexError:
            continue

    return ranked_list



def compute_kernel_similarity(g_train,g_test):
    #sp_kernel = WeisfeilerLehmanOptimalAssignment(n_iter=1, normalize=False)
    sp_kernel = CustomKernelMatrix()

    sp_kernel.fit_transform([g_train])
    sp_kernel.transform([g_test])
    return sp_kernel.transform([g_test])



def join_rec(dict_results, k,recType):
    cut_rec = dict_results[0:k]
    combined_list = []
    for elem in cut_rec:
        rec_graph = elem[0].split(' ')
        if recType == 'class':
           combined_list.extend(rec_graph[0:1])
        elif recType == 'struct':
           combined_list.extend(rec_graph[1:-1])

    return combined_list

def join_rec_2(dict_results):
    #cut_rec = dict_results[0:k]
    combined_list = []
    for elem in dict_results:
        combined_list = elem[0].split(' ')
        print(combined_list)
    return combined_list





def get_recommendations(train_preprocessed, train_data,test_context, result_file,n_classes,n_items,size,rec_type):


        with open(result_file, 'a', encoding='utf8', errors='ignore') as res:

                with open(test_context, 'r', errors='ignore', encoding='utf-8') as f:

                    #res.write(os.path.basename(test_context)+',')
                    lenght = len(f.readlines())
                    #print(os.path.basename(test_context))
                    test_preprocessed, test_data, test_labels = enrich_data(test_context)


                    #test_preprocessed=preprocessing(test_context)
                    # Extract vocabulary
                    vocab = get_vocab(train_preprocessed, test_preprocessed)
                    G_train_nx = create_graphs_of_words(train_preprocessed, vocab, 3)


                    G_test_nx = create_graphs_of_words(test_preprocessed, vocab, 3)
                    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
                    G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))
                    start = time.time()

                    for i in range(0, lenght):
                            results = compute_recommendations(G_train, train_data, G_test, i, size)
                            rec_graph = join_rec(results, n_classes, rec_type)
                    end = time.time()
                    enlapsed = end - start


                    print("Rec time: ", enlapsed)
                    if rec_type == "class":
                        gt_data = get_gt_classes(test_context)
                    if rec_type == "struct":
                        label, gt_data = load_file(test_context)

                    if gt_data:
                        rec_graph = set(rec_graph)
                        rec_graph = list(rec_graph)[0:n_items]
                        list_gt_global = []
                        for gt in gt_data:
                            #print("recommended ", rec_graph)
                            gt_graph = gt.split(' ')
                            #print(gt_graph)
                            if rec_type == 'class':
                                list_gt_global = gt_data
                            elif rec_type == 'struct':
                                list_gt_global.extend(gt_graph[1:-1])


                        #list_gt_global = list(set(list_gt_global))
                        #print('rec ', rec_graph)


                        print('recommended operations ', rec_graph)
                        #present_recommendations(rec_graph)






                        # dict_terms = mapping_vce_terms("C:\\Users\\claud\\OneDrive\\Desktop\\Grakel\\Grakel\\unique_values.txt")
                        # for rec in rec_graph:
                        #     print("new rec list", dict_terms.get(rec))



                        #res.write(' '.join(rec_graph)+',')
                        #res.write(' '.join(gt_graph))
                        if list_gt_global:

                            ### exact match ###
                            succ_std = success_rate(rec_graph, list_gt_global, 1)
                            print('success rate ', success_rate(rec_graph, list_gt_global, 1))
                            pr_std = precision(rec_graph, list_gt_global)
                            print('precision ', precision(rec_graph, list_gt_global))
                            rec_std = recall(rec_graph, list_gt_global)
                            print('recall ', recall(rec_graph, list_gt_global))
                            if pr_std == 0.0 or rec_std == 0.0:
                                f1_std = 0.0
                            else:
                                f1_std = 2 * (pr_std * rec_std) / (pr_std + rec_std)
                            print('f1 ', f1_std)


                            res.write(str(pr_std) + ',' + str(rec_std) + ','+str(f1_std)+','+str(succ_std) +',' + str(enlapsed)+
                                      '\n')



def read_file_as_list(file):
    list_keys = []
    with open(file, 'r') as f:
        for k in f:
            #print(k.strip())
            list_keys.append(k.strip().replace('\n',''))
    return list_keys










