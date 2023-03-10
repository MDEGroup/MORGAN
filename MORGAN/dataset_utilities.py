import os

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import networkx as nx
from networkx import degree_centrality, density
import re
from dinstance_measures import scikit_cosine, levenshtein_ratio_and_distance
#from ranker import glove_sematic_sim
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from jinja2 import Environment, FileSystemLoader



def preprocess_term(term):
    return term.split(',')[0].split(' ')[0].replace('(', '').replace(')', '')

def get_attributes_from_metaclass(metaclass):
    list_attrs= metaclass.split(' ')[1:-1]
    list_results=[]
    for attr in list_attrs:
        list_results.append(attr.split(',')[0].replace('(', ''))

    return list_results

def get_synonyms_attributes(labels,onto_path, train_list):
    list_dict_similar = {}
    for label, t in zip(labels, train_list):
        list_attr = get_attributes_from_metaclass(t)
        for elem in list_attr:
            if label == 'build':
                similar_terms = get_similar_items(onto_path+'top_10_build_attrs.csv', elem)
                tuple_key = (elem, label)
                list_dict_similar.update({tuple_key : similar_terms})
                #print('added synonym for build')
            elif label == 'tex':
                similar_terms = get_similar_items(onto_path+'top_10_tex_attrs.csv', elem)
                tuple_key=(elem, label)
                list_dict_similar.update({tuple_key : similar_terms})
                #print('added synonym for tex')

            elif label == 'office':
                similar_terms = get_similar_items(onto_path+'top_10_office_attrs.csv', elem)
                tuple_key=(elem, label)
                list_dict_similar.update({tuple_key : similar_terms})
                #print('added synonym for office')
            elif label == 'uml':
                similar_terms = get_similar_items(onto_path + 'top_10_uml_attrs.csv', elem)
                tuple_key = (elem, label)
                list_dict_similar.update({tuple_key: similar_terms})
                #print('added synonym for uml')
            #
            elif label == 'req':
                similar_terms = get_similar_items(onto_path + 'top_10_req_attrs.csv', elem)
                tuple_key = (elem, label)
                list_dict_similar.update({tuple_key: similar_terms})
                #print('added synonym for uml')

            elif label == 'sql':
                similar_terms = get_similar_items(onto_path + 'top_10_sql_attrs.csv', elem)
                tuple_key = (elem, label)
                list_dict_similar.update({tuple_key: similar_terms})
                #print('added synonym for sql')
            elif label == 'petri':
                similar_terms = get_similar_items(onto_path + 'top_10_petri_attrs.csv', elem)
                tuple_key = (elem, label)
                list_dict_similar.update({tuple_key: similar_terms})
                #print('added synonym for petri')

            elif label == 'conf':
                similar_terms = get_similar_items(onto_path + 'top_10_conf_attrs.csv', elem)
                tuple_key = (elem, label)
                list_dict_similar.update({tuple_key: similar_terms})
                #print('added synonym for conf')

    return list_dict_similar




def get_sysnonyms_recommeded_items(list_item,labels, onto_path, type_rec):
    list_dict_similar = {}
    l= labels[0]
    if type_rec=="struct":
        for item in list_item:
            item = item.split(',')[0].replace('(','')
            if l == "tex":
                similar_terms = get_similar_items(onto_path+'top_10_tex_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "build":
                similar_terms = get_similar_items(onto_path+'top_10_build_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "req":
                similar_terms = get_similar_items(onto_path+'top_10_req_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "sql":
                similar_terms = get_similar_items(onto_path+'top_10_sql_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "petri":
                similar_terms = get_similar_items(onto_path + 'top_10_petri_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "office":
                similar_terms = get_similar_items(onto_path + 'top_10_office_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "uml":
                similar_terms = get_similar_items(onto_path + 'top_10_uml_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "conf":
                similar_terms = get_similar_items(onto_path + 'top_10_conf_attrs.csv', item)
                list_dict_similar.update({item: similar_terms})
    if type_rec == "class":
        for item in list_item:
            item = item.split(',')[0].replace('(', '')
            if l == "tex":
                similar_terms = get_similar_items(onto_path + 'top_10_tex_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "build":
                similar_terms = get_similar_items(onto_path + 'top_10_build_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "req":
                similar_terms = get_similar_items(onto_path + 'top_10_req_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "sql":
                similar_terms = get_similar_items(onto_path + 'top_10_sql_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "petri":
                similar_terms = get_similar_items(onto_path + 'top_10_petri_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "office":
                similar_terms = get_similar_items(onto_path + 'top_10_office_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "uml":
                similar_terms = get_similar_items(onto_path + 'top_10_uml_classes.csv', item)
                list_dict_similar.update({item: similar_terms})
            if l == "conf":
                similar_terms = get_similar_items(onto_path + 'top_10_conf_classes.csv', item)
                list_dict_similar.update({item: similar_terms})


        #tuple_key = (elem, label)

    return list_dict_similar




def get_synonyms_class(labels,onto_path, train_list):
    list_dict_similar = {}
    for label, elem in zip(labels, train_list):
        elem = preprocess_term(elem)
        if label == 'build':
            similar_terms = get_similar_items(onto_path+'top_10_build_classes.csv', elem)
            tuple_key = (elem, label)
            list_dict_similar.update({tuple_key : similar_terms})
            #print('added synonym for build')
        elif label == 'tex':
            similar_terms = get_similar_items(onto_path+'top_10_tex_classes.csv', elem)
            tuple_key=(elem, label)
            list_dict_similar.update({tuple_key : similar_terms})
            #print('added synonym for tex')

        elif label == 'office':
            similar_terms = get_similar_items(onto_path+'top_10_office_classes.csv', elem)
            tuple_key=(elem, label)
            list_dict_similar.update({tuple_key : similar_terms})
            #print('added synonym for office')
        elif label == 'uml':
            similar_terms = get_similar_items(onto_path + 'top_10_uml_classes.csv', elem)
            tuple_key = (elem, label)
            list_dict_similar.update({tuple_key: similar_terms})
            #print('added synonym for uml')

        elif label == 'req':
            similar_terms = get_similar_items(onto_path + 'top_10_req_classes.csv', elem)
            tuple_key = (elem, label)
            list_dict_similar.update({tuple_key: similar_terms})
            #print('added synonym for uml')
        elif label == 'sql':
            similar_terms = get_similar_items(onto_path + 'top_10_sql_classes.csv', elem)
            tuple_key = (elem, label)
            list_dict_similar.update({tuple_key: similar_terms})
            #print('added synonym for sql')
        elif label == 'petri':
            similar_terms = get_similar_items(onto_path + 'top_10_petri_classes.csv', elem)
            tuple_key = (elem, label)
            list_dict_similar.update({tuple_key: similar_terms})
            #print('added synonym for petri')

        elif label == 'conf':
            similar_terms = get_similar_items(onto_path + 'top_10_conf_classes.csv', elem)
            tuple_key = (elem, label)
            list_dict_similar.update({tuple_key: similar_terms})
            #print('added synonym for conf')

    return list_dict_similar


def read_csv_as_dict(csv_file):
    dict_results = {}
    df_results = pd.read_csv(csv_file, sep=';')
    # print(df_results)
    for key, value in zip(df_results['class'], df_results['sim_items']):
        dict_results.update({key: value})

    return dict_results


def get_similar_items(top_items,term):
    dict_ten_items = read_csv_as_dict(top_items)
    cleaned = str(dict_ten_items.get(term)).strip().replace(' ', '').replace('(', '').replace(')', '').replace('.',
                                                                                                             '').replace(
        "'", '')
    synonyms = ''.join([i for i in cleaned if not i.isdigit()])
    list_synonyms = synonyms.split(',')
    list_synonyms.pop(-1)
    return list_synonyms

def augment_rec_items(origin, similar_items, type_rec):
    augmented_rec=[]
    augmented_rec.extend(origin)
    for item in origin:
        if type_rec == "struct":
            cleaned_item = item.split(',')[0].replace('(', '')
        if type_rec == "class":
            cleaned_item = preprocess_term(item)
        if similar_items.get(cleaned_item):
            top_attr = similar_items.get(cleaned_item)[0]
            if type_rec == "struct":
                top_attr_format = '(' + top_attr + ',' + item.split(',')[1] + ',' + item.split(',')[2]
                augmented_rec.append(top_attr_format)
            if type_rec == "class":
                augmented_rec.append(top_attr)
    return augmented_rec

def augment_attr_data(origin, labels, similar_items):
    for item, label in zip(origin,labels):
        cleaned_item = item.split(',')[0].replace('(','')
        if similar_items.get((cleaned_item, label)):
            top_attr = similar_items.get((cleaned_item,label))[0]
            top_attr_format='('+top_attr+','+item.split(',')[1]+','+item.split(',')[2]
            origin.append(top_attr_format)
    return origin

def augment_input_data(origin,labels, similar_items, similar_attrs):
    results = []
    for item, label in zip(origin, labels):
        cleaned_item = preprocess_term(item)
        if similar_items.get((cleaned_item,label)):
            top_syn = similar_items.get((cleaned_item,label))[0]
            attrs = item.split(' ')[1:-1]
            augmented_attrs = augment_attr_data(attrs,labels,similar_attrs)
            augmented_origins = [cleaned_item] + augmented_attrs
            augmented = [top_syn] + augmented_attrs
            string_augmented = " ".join(augmented)
            string_augmented_origin= " ".join(augmented_origins)
            results.append(string_augmented)
            results.append(string_augmented_origin)

    return results

def create_tuple_list(label_list, data_list):
    list_tuple=[]
    for l, d in zip(label_list, data_list):
        tuple_data = l, d
        list_tuple.append(tuple_data)

    return list_tuple

def split_dataset(filename):
    labels = []
    test_docs = []
    train_docs = []

    with open(filename, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if line.find('\t') != -1:
                content = line.split('\t')
                labels.append(content[0])
                graph_tot = content[1].split(" ")[:-1]
                size = (len(graph_tot * 2) / 3)
                split_test = graph_tot[int(size): -1]
                string_train = ' '.join([str(elem) for elem in graph_tot])
                string_test = ' '.join([str(elem) for elem in split_test])
                train_docs.append(string_train)
                test_docs.append(string_test)

    return train_docs, test_docs, labels


def load_file(filename):
    labels = []
    docs = []

    with open(filename,'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if line.find('\t')!=-1:
                content = line.split('\t')


                if len(content) > 0:
                    labels.append(content[0])
                    docs.append(content[1])


    return labels, docs


def get_gt_classes(filename):
    labels = []
    docs = []
    try:
        with open(filename,'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                if line.find('\t')!= -1:
                    content = line.split('\t')
                    labels.append(content[0])
                    docs.append(content[1].split(' ')[0])
    except:
        print(filename)
        return None
    return docs



def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    ## nlp here ##
    #string = re.sub(r"\(", "", string)
    #string = re.sub(r"\)", "", string)
    ##


    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def find_unique_values(train_data):
    with open("unique_values.txt", "w", encoding="utf8", errors="ignore") as res:

        for train in train_data:
            attrs = train.split(" ")
            unique = set(attrs)

            for u in unique:
                res.write(u + "\n")








def mapping_vce_terms(file_attrs):
    with open(file_attrs) as f:
        lines = f.read().splitlines()

    list_set = set(lines)
    # convert the set to the list
    unique_list = (list(list_set))
    volvo_terms = []

    dict_terms = {}
    for key, term in zip(unique_list, volvo_terms):
        dict_terms.update({key: term})




    return dict_terms



def preprocessing(docs):
    preprocessed_docs = []
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()


    for doc in docs:
        clean_doc = clean_str(doc)

        new_values = []

        #print(new_values)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
        #preprocessed_docs.append([wordnet_lemmatizer.lemmatize(w) for w in clean_doc])
    return preprocessed_docs


def get_vocab_train(train_docs):
    vocab = dict()
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def get_vocab(train_docs, test_docs):
    vocab = dict()

    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def create_graphs_of_words(docs, vocab, window_size):
        graphs = list()

        for idx, doc in enumerate(docs):
            G = nx.Graph()
            for i in range(len(doc)):
                if doc[i] not in G.nodes():
                    G.add_node(doc[i])
                    G.nodes[doc[i]]['label'] = vocab[doc[i]]
            for i in range(len(doc)):
                for j in range(i + 1, i + window_size):
                    if j < len(doc):
                        G.add_edge(doc[i], doc[j])

            graphs.append(G)

        return graphs



def convert_string_to_list(list_element):
    str_format=''
    return str_format.join(list_element)


def enrich_gt_data(gt_context):
    y_train, train_data = load_file(gt_context)

    #similar_class = get_synonyms_class(y_train, 'C:/Users/claudio/PycharmProjects/Grakel/Datasets/Ontologies/',
                                      # train_data)
    #similar_attributes = get_synonyms_attributes(y_train,  'C:/Users/claudio/PycharmProjects/Grakel/Datasets/Ontologies/',                                                 train_data)
    #train_augmented = augment_input_data(train_data, y_train, similar_class, similar_attributes)
    #print(train_augmented)

    return train_data


def enrich_data(train_context):

    y_train, train_data = load_file(train_context)
    #print('len original data', len(train_data))
   # similar_class = get_synonyms_class(y_train, 'C:/Users/claudio/PycharmProjects/Grakel/Datasets/Ontologies/', train_data)

    #similar_attributes = get_synonyms_attributes(y_train, 'C:/Users/claudio/PycharmProjects/Grakel/Datasets/Ontologies/', train_data)

    #train_augmented = augment_input_data(train_data, y_train, similar_class, similar_attributes)
    #print('augmented data ', len(train_augmented))



    train_preprocessed = preprocessing(train_data)
    find_unique_values(train_data)




    return train_preprocessed, train_data, y_train


def preprocess_test_data(test_context):
    y_test, test_data = load_file(test_context)
    test_preprocessed = preprocessing(test_data)
    return test_preprocessed



def success_rate_sim(predicted, actual,sim_func, threshold):
    if actual and predicted:
        match = 0

        if sim_func == 'cosine':
            for pred in predicted:
                list_tuple = []
                dict_sim = {}
                for gt in actual:
                    list_tuple.append((gt, scikit_cosine(pred, gt)[0][0]))
                ranked_list = sorted(list_tuple, key=lambda tup: tup[1], reverse=True)
                dict_sim.update({pred: ranked_list})
                values = dict_sim.values()
                iterator = iter(values)
                first_elem = next(iterator)
                first_item_sim = first_elem[0][1]
                if first_item_sim > threshold:
                    match +=1

        if sim_func == 'lev':
            for pred in predicted:
                list_tuple = []
                dict_sim = {}
                for gt in actual:
                    list_tuple.append((gt, levenshtein_ratio_and_distance(pred,gt)))
                ranked_list = sorted(list_tuple, key=lambda tup: tup[1], reverse=True)
                dict_sim.update({pred: ranked_list})
                values = dict_sim.values()
                iterator = iter(values)
                first_elem = next(iterator)
                first_item_sim = first_elem[0][1]

                if first_item_sim > threshold:
                    match +=1

        if match >= 1:
            return 1
        else:
            return 0

    else:
        return 0


def success_rate(predicted, actual, n):
    if actual:
        match = [value for value in predicted if value in actual]

        if len(match) >= n:
            return 1
        else:
            return 0
    else:
        return 0


def precision_sim(predicted,actual, sim_func, threshold):
    if actual and predicted:
        true_p= 0
        false_p = 0
        if sim_func=='cosine':
            for pred in predicted:
                list_tuple = []
                dict_sim = {}
                for gt in actual:
                    list_tuple.append((gt, scikit_cosine(pred,gt)[0][0]))
                ranked_list = sorted(list_tuple, key=lambda tup: tup[1], reverse=True)
                dict_sim.update({pred: ranked_list})
                values = dict_sim.values()
                iterator = iter(values)
                first_elem = next(iterator)
                first_item_sim = first_elem[0][1]

                if first_item_sim > threshold:
                    true_p += 1
                else:
                    false_p += 1
        elif sim_func == 'lev':
            for pred in predicted:
                list_tuple = []
                dict_sim = {}
                for gt in actual:
                    list_tuple.append((gt, levenshtein_ratio_and_distance(pred, gt)))
                ranked_list = sorted(list_tuple, key=lambda tup: tup[1], reverse=True)
                dict_sim.update({pred: ranked_list})
                values = dict_sim.values()
                iterator = iter(values)
                first_elem = next(iterator)
                first_item_sim = first_elem[0][1]

                if first_item_sim > threshold:
                    true_p += 1
                else:
                    false_p += 1


        #print('precision stats', true_p,false_p)
        return (true_p / (true_p + false_p))*100
    else:
        return 0


def precision(predicted,actual):
    if actual and predicted:
        true_p = len([value for value in predicted if value in actual])
        false_p = len([value for value in predicted if value not in actual])
        return (true_p / (true_p + false_p))*100
    else:
        return 0

def recall_sim(predicted,actual, sim_func, threshold):
    if actual and predicted:

        true_p = 0
        false_n = 0
        if sim_func == 'cosine':
            for gt in actual:
                list_tuple = []
                dict_sim = {}
                for pred in predicted:
                    list_tuple.append((gt, scikit_cosine(pred, gt)[0][0]))
                ranked_list = sorted(list_tuple, key=lambda tup: tup[1], reverse=True)
                dict_sim.update({gt: ranked_list})
                values = dict_sim.values()
                iterator = iter(values)
                first_elem = next(iterator)
                first_item_sim = first_elem[0][1]

                if first_item_sim > threshold:
                    true_p += 1
                else:
                    false_n += 1

        if sim_func == 'lev':
            for gt in actual:
                list_tuple = []
                dict_sim = {}
                for pred in predicted:
                    list_tuple.append((gt, levenshtein_ratio_and_distance(pred, gt)))
                ranked_list = sorted(list_tuple, key=lambda tup: tup[1], reverse=True)
                dict_sim.update({gt: ranked_list})
                values = dict_sim.values()
                iterator = iter(values)
                first_elem = next(iterator)
                first_item_sim = first_elem[0][1]

                if first_item_sim > threshold:
                    true_p += 1
                else:
                    false_n += 1
        #print('recall stats',true_p, false_n)
        return (true_p/(true_p + false_n))*100
    else:
        return 0


def recall(predicted,actual):
    if actual and predicted:
        # true_p = len([value for value in predicted if value in actual])
        false_n = len([value for value in actual if value not in predicted])
        true_p = len([value for value in predicted if value in actual])
        return (true_p/(true_p + false_n))*100
    else:
        return 0


def format_dict(dict):

    out_string = ""
    i = 0
    for key, value in dict.items():
        out_string += str(key)+":"+str(value)+"#"
    return out_string



def compute_graph_metrics(graph_list, file_out):

    for graph in graph_list:
        #print("Graph size:", graph.size())
        file_out.write(str(graph.size())+',')
        #print("Graph order: ", graph.order())
        file_out.write(str(graph.order()) + ',')
        degree_sequence = [str(d) for n, d in graph.degree()]
        #print("Graph degree sequence:", degree_sequence)
        file_out.write(":".join(degree_sequence)+",")
        #print("Graph density: ", density(graph))
        file_out.write(str(density(graph)) + ',')
        centrality =  degree_centrality(graph)
        #print("Graph centrality: ", centrality)
        file_out.write(format_dict(centrality))
        file_out.write("\n")

        file_out.flush()


# def compute_semantic_similarity(folder, out_file):
#     list_docs = []
#     stop_words = []
#
#     with open(out_file, "w", encoding="utf8", errors='ignore') as res:
#         for file in os.listdir(folder):
#             f = open(folder+file, 'r', encoding="utf8", errors="encoding")
#             list_docs.append(f.read().rstrip())
#
#         similarity_matrix=[]
#         for i in range(0, len(list_docs)):
#             doc = list_docs.pop(i)
#
#             list_similarities = list(1-glove_sematic_sim(doc,list_docs, stop_words))
#             list_similarities.insert(i,1.0)
#             res.write(','.join(str(s) for s in list_similarities)+'\n')
#             similarity_matrix.append(list_similarities)
#             list_docs.insert(i,doc)
#
#
#     return similarity_matrix

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def read_similarity_matrix(file_path):

    matrix=[]
    f= open(file_path, "r", encoding="utf8", errors="ignore")
    lines = f.readlines()
    for l in lines:
        l = l.strip()
        list_sim= l.split(",")
        list_float = [float(s) for s in list_sim]
        matrix.append(list_float)

    return matrix



def present_recommendations(recommendations):
    max_score = 100
    test_name = "Python Challenge"
    students = [
        {"name": "Sandrine", "score": 100},
        {"name": "Gergeley", "score": 87},
        {"name": "Frieda", "score": 92},
    ]

    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("message.txt")

    for student in students:
        filename = f"message_{student['name'].lower()}.txt"
        content = template.render(
            student,
            max_score=max_score,
            test_name=test_name
        )
        with open(filename, mode="w", encoding="utf-8") as message:
            message.write(content)
            print(f"... wrote {filename}")







