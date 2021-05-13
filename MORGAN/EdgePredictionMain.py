from grakel import Graph
from Grakel.dataset_utilities import load_file, aggregate_cluster_files, clean_files
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.datasets import fetch_dataset
import os



wl_kernel = WeisfeilerLehman(base_graph_kernel=VertexHistogram)

def preprocess_dataset():
    dataset_path = 'C:/Users/claudio/Desktop/ecore_new/office/'
    out_path = 'C:/Users/claudio/Desktop/train_docs/'
    filename = "office_cluster.txt"
    aggregate_cluster_files(dataset_path, out_path, filename)
    clst_path = 'C:/Users/claudio/Desktop/train_docs/office_cluster.txt'
    classname='Office'
    clean_files(clst_path,'office.txt', classname+'.txt')

def load_train(pathFile):
    list_train_graphs = []
    train_data, train_labels = load_file(pathFile)

    #print(train_data, train_labels)
    print(train_labels)


    for t in train_data:
        print(t)
        node_labels = {}
        adj_matrix = []

        for elem, i in zip(t.split(" "), range(0, len(train_labels))):
            print([1]*len(train_labels))
            adj_matrix.append([1]*len(train_labels))
            #print(elem, i)
            node_labels.update({i: elem})
            g = Graph(initialization_object=adj_matrix, node_labels=node_labels)
            #print(g.get_labels())
        list_train_graphs.append(g)

    return list_train_graphs, train_labels

# file1 = 'C:/Users/claudio/Desktop/train_docs_cleaned/MARTE.txt'
# file2 = 'C:/Users/claudio/Desktop/train_docs_cleaned/MSOfficeExcel.txt'
# graph_1, labels1=load_train(file1)
# graph_2, lables2=load_train(file2)

# for g in graph_2:
#     print(g.get_labels())
#
#
#
# train_labels = labels1 + lables2
# list_graph = graph_1 + graph_2


# train_office = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]
# office_labels = {0: 'A', 1: 'B', 2: 'C', 3 : 'D' ,4: 'E' }
# graph_office = Graph(initialization_object=train_office, node_labels=office_labels)
#
# train_marte = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
# marte_labels = {0: 'X', 1: 'F', 2: 'Y' }
# graph_marte = Graph(initialization_object=train_marte, node_labels=marte_labels)
#
# print(graph_marte.get_labels())
#
#
# test_adjacency = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]
# test_labels = {0: 'A', 1: 'B', 2: 'C', 3 : 'D' ,4: 'E' }
# test = Graph(initialization_object=test_adjacency, node_labels=test_labels)
# print(test.get_labels())
#
# K_train = wl_kernel.fit_transform([graph_marte, graph_office])
# K_test = wl_kernel.transform([test])
#
# train_labels=['Marte', 'Office']
#
# from sklearn.svm import SVC
# clf = SVC(kernel='precomputed')
# clf.fit(K_train, train_labels)
#
# y_pred = clf.predict(K_test)
# print(y_pred)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath

MUTAG = fetch_dataset("MUTAG", verbose=False)
G = MUTAG.data
y = MUTAG.target

# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)

# Uses the shortest path kernel to generate the kernel matrices
gk = ShortestPath(normalize=True)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")