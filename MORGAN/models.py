from grakel.kernels import WeisfeilerLehman
from grakel.utils import graph_from_networkx
import os
from dataset_utilities import load_file,preprocessing,get_vocab, create_graphs_of_words


def retrieve_recommendations(train_path,test_path,n,size):

    train_data, y_train = load_file(train_path)
    test_data, y_test = load_file(test_path)
    ranked_list = ()
    # Read and pre-process test data

    train_preprocessed = preprocessing(train_data)
    test_preprocessed = preprocessing(test_data)

    # Extract vocabulary
    vocab = get_vocab(train_preprocessed, test_preprocessed)
    G_train_nx = create_graphs_of_words(train_preprocessed, vocab, 3)
    G_test_nx = create_graphs_of_words(test_preprocessed, vocab, 3)
    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
    G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))


    list_sim = []
    for g, rec in zip(G_train, train_data):
        try:
            if len(G_test[n]) > 0:
                if size == 1:
                    tot = 3
                elif size == 2:
                    tot = (len(G_test[n]) * size) / 3

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



def compute_kernel_similarity(g_train,g_test):
    sp_kernel = WeisfeilerLehman(n_iter=1, normalize=False)
    sp_kernel.fit_transform([g_train])
    sp_kernel.transform([g_test])
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



def retrieve_similar_class(train_context, test_context,gt_context, result_file,n_classes,n_items,size,recType):

    with open(result_file, 'a', encoding='utf8', errors='ignore') as res:
        with open(test_context, 'r', errors='ignore', encoding='utf-8') as f:
            res.write(os.path.basename(test_context)+'\n')
            lenght=len(f.readlines())
            print(lenght)
            if lenght < 10:
                for i in range(0, lenght):
                        results = retrieve_recommendations(train_context, test_context, i, size)
                        rec_graph = join_rec(results, n_classes,recType)
                gt_data, gt_label = load_file(gt_context)
                for gt in gt_data:
                    rec_graph = rec_graph[0:n_items]
                    print("recommended ", rec_graph)
                    gt_graph = gt.split(' ')
                    if recType == 'class':
                        gt_graph = gt_graph[0:1]
                    elif recType == 'struct':
                        gt_graph = gt_graph[1:-1]
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















