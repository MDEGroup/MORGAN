from sklearn.metrics import accuracy_score, precision_recall_fscore_support, top_k_accuracy_score
from grakel.kernels import WeisfeilerLehman, ShortestPath, HadamardCode, NeighborhoodHash
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import operator
from grakel import Graph

import re
from grakel.utils import cross_validate_Kfold_SVM
from grakel.datasets import fetch_dataset
import numpy as np
from dataset_utilities import *



def remove_labels(string):
    label_list_ecore=['Petri', 'Req', 'Sql', 'Uml','Office','Build', 'Conf', 'Tex']
    #label_list_xmi =['apache', 'parser', 'build','sdk','spring','sql','testing','websever']
    for l in label_list_ecore:
        string= string.strip().replace(l, '').replace('\t','')
    return string

def match_items(result,groundt):
    list_classes = []
    count = 0
    with open(result, 'r', encoding='utf-8', errors='ignore') as res:
        with open(groundt, 'r', encoding='utf-8', errors='ignore') as gt:
            list_gt = gt.readlines()
            list_res = res.readlines()
            for test in list_gt:
                #print(test.strip())
                for out in list_res:
                    cleaned = remove_labels(test)
                    if out.find(cleaned) != -1:
                        result = ''.join([i for i in out.strip() if not i.isdigit()])
                        list_classes.append(result)
    list_classes = list(dict.fromkeys(list_classes))
    return list_classes, len(list_gt)
                    # if cleaned == out:
                    #     print('match')




def retrieve_recommendations( train_path,test_path,n):

    train_data, y_train = load_file(train_path)
    test_data, y_test = load_file(test_path)
    ranked_list=()
    # Read and pre-process test data

    train_preprocessed = preprocessing(train_data)
    test_preprocessed = preprocessing(test_data)

    # Extract vocabulary
    vocab = get_vocab(train_preprocessed, test_preprocessed)
    G_train_nx = create_graphs_of_words(train_preprocessed, vocab, 3)
    G_test_nx = create_graphs_of_words(test_preprocessed, vocab, 3)
    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
    G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))

    #y_pred = choose_kernel_model(G_train, G_test, y_train, model='weis', classfier='svm')


    list_sim = []
    for g, rec in zip(G_train, train_data):
        try:
            if len(G_test[n]) > 0:
                tot = (len(G_test[n]) * 2) / 3
                if tot > 0:
                    sim = compute_kernel_similarity(g, G_test[n][0:int(tot)])
                    if sim[0][0] > 0:
                        tuple_g = rec, sim[0][0]
                        list_sim.append(tuple_g)
                    ranked_list = sorted(list_sim, key=lambda tup: tup[1], reverse=True)
            else:
                continue
        except IndexError:
            continue

    return ranked_list


def compute_metrics(predicted, actual):

        precision, recall, fscore , true = precision_recall_fscore_support(actual,predicted, zero_division=1, average='weighted')
        accuracy = accuracy_score(actual, predicted)
        #Evaluate the predictions
        #print("Accuracy:", accuracy_score(predicted, actual))

        print("Precision:", precision)
        print("Recall:", recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("F1: ", f1)

        return precision, recall, f1




    # with open(file, 'a', encoding='utf-8', errors='ignore') as res:
    #     #res.write('precision, recall, fmeasure \n')
    #     res.write(str(precision) + "," + str(recall)+',' + str(f1) + '\n')

def choose_kernel_model(train, test, y_train, model, classfier):

    if model == "weis":
        gk = WeisfeilerLehman(n_iter=1, normalize=False)
    elif model == 'had':
        gk = HadamardCode(n_iter=1, normalize=False)
    elif model == 'hash':
        gk = NeighborhoodHash(normalize=False)

    # Construct kernel matrices
    K_train = gk.fit_transform(train)
    K_test = gk.transform(test)

    # Train an SVM classifier and make predictions
    if classfier== 'svm':
        clf = SVC(kernel='precomputed')
    elif classfier =='rf':
        clf =RandomForestClassifier()
    elif classfier == 'mlp':
        clf = MLPClassifier()
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)

    return y_pred

def compute_kernel_similarity(g_train,g_test):
    #sp_kernel = ShortestPath(normalize=True)
    sp_kernel = WeisfeilerLehman(n_iter=1, normalize=False)
    sp_kernel.fit_transform([g_train])

    sp_kernel.transform([g_test])
    #print(sp_kernel.transform([g_test]))
    return sp_kernel.transform([g_test])

def success_rate(predicted, actual, n):
    if actual:
        match = [value for value in predicted if value in actual]
        if len(match) >= n:
            return 1
        else:
            return 0
    else:
        return 0

def precision(predicted,actual):
    if actual and predicted:
        true_p = len([value for value in predicted if value in actual])
        false_p = len([value for value in predicted if value not in actual])

        return (true_p / (true_p + false_p))*100
    else:
        return 0


def recall(predicted,actual):
    if actual and predicted:
        true_p = len([value for value in predicted if value in actual])
        false_n = len([value for value in actual if value not in predicted])
        return (true_p/(true_p + false_n))*100
    else:
        return 0


def join_rec(dict_results, k):
    cut_rec = dict_results[0:k]
    combined_list = []
    for elem in cut_rec:
        rec_graph = elem[0].split(' ')
        #combined_list.extend(rec_graph[1:-1])
        combined_list.extend(rec_graph[0:1])


    return combined_list



def retrieve_similar_class(train_context, test_context,gt_context, result_file):

    with open(result_file, 'a', encoding='utf8', errors='ignore') as res:
        with open(test_context, 'r', errors='ignore', encoding='utf-8') as f:
            res.write(os.path.basename(test_context)+'\n')
            #res.write('precision, recall, fmeasure, succ_rate  \n')
            lenght=len(f.readlines())
            print(lenght)
            if lenght < 10:
                for i in range(0, lenght):
                        results = retrieve_recommendations(train_context, test_context, i)
                        rec_graph = join_rec(results, 4)

                gt_data, gt_label = load_file(gt_context)
                for gt in gt_data:

                    rec_graph = rec_graph[0:4]
                    print("recommended ", rec_graph)
                    gt_graph = gt.split(' ')
                    #gt_graph = gt_graph[1:-1]
                    gt_graph = gt_graph[0:1]
                    print("gt class ", gt_graph)
                    pr = precision(rec_graph,gt_graph)
                    print(pr)
                    rec= recall(rec_graph, gt_graph)
                    if pr == 0.0 or rec == 0.0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (pr * rec) / (pr + rec)

                    succ = success_rate(rec_graph,gt_graph, 1)

                    res.write(str(pr)+ ','+ str(rec) + ','+str(f1)+','+str(succ) + '\n')















