import re
from nltk.stem.porter import PorterStemmer
from nltk.metrics import edit_distance, jaccard_distance
import networkx as nx
import matplotlib.pyplot as plt
from grakel.utils import graph_from_networkx
import shutil, os
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, math
from collections import Counter
from textdistance import levenshtein

WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def compute_edit_distance(gt_list, rec_list):
    with open('cosine_results_3.txt','w',encoding='utf-8', errors='ignore') as res:
        #res.write('pair,sim \n')
        for gt in gt_list:
            for rec in rec_list:
                #print(gt, rec)
                #res.write(gt+': '+rec+ '\n')
                lenght = len(rec) + len(gt)
                # vect1=text_to_vector(gt)
                # vect2 = text_to_vector(rec)
                value= levenshtein(gt,rec)
                #res.write(str(get_cosine(vect1,vect2))+ '\n')
                res.write(str(1 - value/lenght) + '\n')
                #print(get_cosine(vect1,vect2))





def get_rec_class(file):
    list_rec = []
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for l in lines:
            if l.find('class') == -1:
                if l.find(' ') != -1:
                    split = l.strip().split(' ')
                    list_rec.append(split[0].replace('(', '').replace("'","").lower())

                else:
                    continue
    return list_rec

def get_gt_class(file):
    list_gt = []
    with open(file,'r',encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        for l in lines:
            if l.find('\t')!= -1:
                split = l.strip().split('\t')
                content = split[1]
                content = content.split(' ')
                list_gt.append(content[0].lower())
    return list_gt



def emf_compare_statistics(file):
    df_emf = pd.read_csv(file, sep='\t')
    print(df_emf.describe())

    boxplot = df_emf.boxplot()
    myFig = plt.figure()
    bp = df_emf.boxplot()
    myFig.savefig("example.png", format="png")

def filter_file(dataset_path, out_path):
    #dataset_path = './single_project/'
    #out_path = './single_project_methods/'

    for file in os.listdir(dataset_path):

        filter_attributes(dataset_path+'/'+file, out_path+'/'+file)


def ten_folder_classes():
    for i in range (1,11):
        cluster_path ='C:/Users/claudio/Desktop/ten_fold_ecore_structure/test'+str(i)+'/'
        #filter_path= 'C:/Users/claudio/Desktop/ten_fold_ecore_structure/raw_test_ecore/'
        filter_path = './test_categories/test_' + str(i) + '/'

        filter_file(cluster_path, filter_path)

        out_gt_path = 'C:/Users/claudio/Desktop/Morgan_conf/C1.1_ecore/gt_'+str(i)+'/'
        out_test_path = 'C:/Users/claudio/Desktop/Morgan_conf/C1.1_ecore/test_'+str(i)+'/'
        # for file in os.listdir(cluster_path):
        #     remove_duplicates(cluster_path+file,filter_path+file)

        for file in os.listdir(filter_path):
            with open(filter_path+file, 'r') as f:
                num = int((len(f.readlines())) / 3)
                split_test_files(filter_path+file, num, file, out_gt_path, out_test_path)


def move_similar_file(src, dest, list_similar):
    for file in os.listdir(src):
        for elem in list_similar:
            c_file = str(file).replace('.txt', '').replace('_result','')
            print(c_file)
            if c_file == str(elem):
                shutil.copy(src+file, dest+file)




def find_similar_model(cat):
    list_similar=[]
    for file in os.listdir(cat):
        with open(cat+'/'+file, 'r') as red:
            clean = red.readlines()[0].strip().split('\t')[0]
            list_similar.append(clean)
    no_dup = list(dict.fromkeys(list_similar))
    return no_dup



def move_files(abs_dirname,N, outpath):
    """Move files into subdirectories."""

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    curr_subdir = None

    for f in files:
        # create new subdir if necessary
        if i % N == 0:
            subdir_name = os.path.join(abs_dirname, str(int((i / N + 1))))
            os.mkdir(outpath+subdir_name)
            curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1




def compute_statistics(file):
    tot_classes=0
    tot_ref =0
    tot_attr=0
    tot_files = 0

    #print(file)
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines=f.readlines()
        #print(len(lines))
        #tot_files = tot_files + 1
        tot_classes= len(lines)+ tot_classes
        for l in lines:
            if l.find('field') != -1:
                tot_ref= tot_ref + 1
            if l.find('method') != -1:
                tot_attr = tot_attr + 1
    #print('tot files', tot_files)
    print('tot class', tot_classes)
    print('tot_ method' , tot_attr)
    print('tot_field ', tot_ref)



def filter_files_by_lines(src, dst):
    for file in os.listdir(src):
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if lines >= 400:
                shutil.copy(src+ file, dst+ file)



def filter_attributes(filename, outfile):
    #labels = []
    #docs = []

    with open(filename,'r', encoding='utf8', errors='ignore') as f:
        with open(outfile,'w', encoding='utf8', errors='ignore') as out:
            for line in f:
                if line.find('\t') != -1:
                    content = line.split('\t')
                    elements =content[1].split(" ")[:-1]
                    if len(elements) > 6:
                        out.write(line)
                        #print(line)
                    #labels.append(content[0])
                    #docs.append(content[1][:-1])






def create_ten_fold_structure(cat_path):
    out_path = './train_root/'
    splitted_path='./split_files/'
    for fold in os.listdir(cat_path):
        for i in range(1, 11):
            for file in os.listdir(splitted_path):
                filename ='./split_files/'+fold+'.txt_'+str(i)+'.txt'


                folder = './train_root/train_partial_' + str(i)+'/'
                print(filename)
                print(folder)


                try:
                    shutil.copy(filename,folder)
                except FileNotFoundError:
                    continue

def select_fileby_size(filepath,src, dst):
    dim = os.path.getsize(filepath)
    print(dim)
    if dim > 0:
        shutil.copy(filepath, dst+str(filepath).replace(src,''))




def create_tuple_list(label_list, data_list):
    list_tuple=[]
    for l, d in zip(label_list, data_list):
        tuple_data = l, d
        list_tuple.append(tuple_data)

    return list_tuple

def mapping_files():
    path= "C:/Users/claudio/Desktop/modisco_models/"
    out_path="C:/Users/claudio/Desktop/train_dataset_xmi/"
    df = pd.read_csv('xmi_list.csv', sep=",")
    #print(set(df['category']))

    for cat in set(df['category']):
        os.mkdir(out_path+cat)

    # for file in os.listdir(path):
    #     #print(file)
    #     for tag,name in zip(df['category'], df['filename']):
    #         #print(str(name), tag)
    #             #print(str(name), tag)
    #             if (str(name).find(str(file).replace(".xmi", "")) != -1):
    #                 shutil.copy(path + file, out_path + tag +'/')




def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile, 2):
        if random.randrange(num):
            continue
        line = aline
    return line

def aggregate_cluster_files(path, outpath,filename):
    with open(outpath+filename, 'wb') as wfd:
        for f in os.listdir(path):
            with open(path+f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def clean_files(clusterFile, outfile, classname):
    with open(outfile, 'w', encoding='utf-8', errors='ignore') as out:
        with open(clusterFile, 'r', encoding='utf-8', errors='ignore') as cluster:
            for line in cluster.readlines():
                splitted= line.split('#')
                out.write(classname.replace('.txt','')+'#'+splitted[1])


def load_original_dataset(filename):

    labels = []
    docs = []

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split('\t')
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels

def split_dataset(filename):
    labels = []
    test_docs = []
    train_docs = []

    with open(filename, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if line.find('\t')!=-1:
                content = line.split('\t')
                labels.append(content[0])
                graph_tot = content[1].split(" ")[:-1]

                size = (len(graph_tot * 2) / 3)
                #split_train = graph_tot[0:int(size)]
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
            if line.find('\t')!= -1:
                content = line.split('\t')
                labels.append(content[0])
                docs.append(content[1][:-1])


    return docs, labels


def load_file_test(filename):
    labels = []
    docs = []

    with open(filename,'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split('\t')
            labels.append(content[0])
            graph_tot = content[1].split(" ")[:-1]

            size = len(graph_tot) / 3
            split_test_first = graph_tot[0: int(size)]
            #split_test_second = graph_tot[int(size): -1]
            string_test = ' '.join([str(elem) for elem in split_test_first])
            docs.append(string_test)




    return docs, labels


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs):
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])

    return preprocessed_docs


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
        sizes = list()
        degs = list()

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


def create_dataset_dict(data, labels):

    dict_list =[]
    data_dict={}



    for train, label in zip(data,labels):

        #print(label, train)
        zip_iterator = zip(label, train)
        data_dict = dict(zip_iterator)
        dict_list.append(data_dict.update({label: train}))
        #print(data_dict)

    return dict_list

def get_graphs_single_eval(path_dataset,path_test,start,end):
    train_data, y_train = load_file(path_dataset)
    test_data, y_test = load_file_test(path_test)

    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)

    train_dict = create_dataset_dict(train_data, y_train)
    test_dict = create_dataset_dict(test_data, y_test)

    # Extract vocabulary
    vocab = get_vocab(train_data, test_data)
    print("Vocabulary size: ", len(vocab))

    # Create graph-of-words representations
    G_train_nx = create_graphs_of_words(train_data, vocab, 3)
    G_test_nx = create_graphs_of_words(test_data, vocab, 3)

    print("Example of graph-of-words representation of document")

    nx.draw_networkx(G_train_nx[0], with_labels=True)
    nx.draw_networkx(G_test_nx[0], with_labels=True)
    plt.show()
    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
    G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))

    print(len(G_test))

    return G_train, G_test, y_train, y_test, train_dict, test_dict



def get_graphs_cross_eval(path):

    train_data, test_data, labels = split_dataset(path)


    train_preprocessed = preprocessing(train_data)
    test_preprocesed = preprocessing(test_data)

    #train_dict = create_dataset_dict(train_data, labels)
    #test_dict = create_dataset_dict(test_data, labels)
    vocab = get_vocab(train_preprocessed, test_preprocesed)
    print("Vocabulary size: ", len(vocab))

    # Create graph-of-words representations
    G_train_nx = create_graphs_of_words(train_preprocessed, vocab, 3)
    G_test_nx = create_graphs_of_words(test_preprocesed, vocab, 3)

    #print("Example of graph-of-words representation of document")

    #nx.draw_networkx(G_train_nx[0], with_labels=True)

    # print("Example of graph-of-words representation of document")
    #
    # nx.draw_networkx(G_train_nx[0], with_labels=True)
    # plt.show()


    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
    G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))

    return G_train, G_test, labels, train_data


def get_original_graphs(path_to_train_set, path_to_test_set ):


    # Read and pre-process train data
    train_data, y_train = load_file(path_to_train_set)
    test_data, y_test = load_file(path_to_test_set)

    # Read and pre-process test data
    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)



    # Extract vocabulary
    vocab = get_vocab(train_data, test_data)
    G_train_nx = create_graphs_of_words(train_data, vocab, 3)
    G_test_nx = create_graphs_of_words(test_data, vocab, 3)


    # print("Example of graph-of-words representation of document")
    #
    # nx.draw_networkx(G_train_nx[0], with_labels=True)
    # plt.show()

    G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
    G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))

    return G_train, G_test, y_train, y_test

def remove_duplicates(src, dst):
    lines_seen = set()  # holds lines already seen
    with open(dst, "w") as output_file:
        for each_line in open(src, "r"):
            if each_line not in lines_seen:  # check if line is not duplicate
                output_file.write(each_line)
                lines_seen.add(each_line)


def split_test_files(root_path, n, filename,gt, test):
    from itertools import zip_longest


    def grouper(n, iterable, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)


    lines_per_file = 300
    smallfile = None

    with open(root_path) as f:

        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            if i==1:
                with open(test+filename+'_{0}.txt'.format(i), 'w') as fout:
                    fout.writelines(g)
            else:
                with open(gt+filename+'_{0}.txt'.format(i), 'w') as fout:
                    fout.writelines(g)

def load_train(train_path, test_path):
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

    return G_train,G_test, test_data
