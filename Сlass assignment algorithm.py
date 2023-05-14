import numpy as np
from numpy import isnan
from kneebow.rotor import Rotor
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy import stats
from sas7bdat import SAS7BDAT



def unique(list1):
    unique_list = []

    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def listsum(list):
    s = 0
    for i in list:
        s += i
    return s

def make_lables(values_clusters_maked, mark, t):
    values_classes_maked = []
    for k in range(len(values_clusters_maked[mark].unique()) + 1):
        values_classes_maked.append([])
    len(values_classes_maked)
    values_clusters_maked.sort_values(by=[i], inplace=True)
    values_clusters_maked.reset_index(drop=True)
    for k in values_clusters_maked.index:
        if values_clusters_maked[i][k] in values_classes_maked[0]:
            values_classes_maked[values_clusters_maked[mark][k] + t][values_classes_maked[0].index(values_clusters_maked[i][k])] += 1
        else:
            values_classes_maked[0].append(values_clusters_maked[i][k])
            for s in range(1, len(values_classes_maked)):
                values_classes_maked[s].append(0)
            values_classes_maked[values_clusters_maked[mark][k] + t][-1] += 1

    # choose classes
    values_classes_maked.append([])
    for k in range(len(values_classes_maked[0])):
        p_class = 0
        max_val = 0
        for s in range(len(values_classes_maked) - 2):
            if values_classes_maked[s + 1][k] > max_val:
                p_class = s
                max_val = values_classes_maked[s + 1][k]
        values_classes_maked[-1].append(p_class)
    return values_classes_maked


with SAS7BDAT('file_name.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()

print(df.columns)


cat = ['TRTPN', 'RACEN', 'SEXN']
num = ['CL', 'BILI','ALP', 'GGT', 'ALT', 'CREAT', 'URATE', 'PHOS', 'GLUC', 'CHOL', 'CK', 'CA',
        'AGE', 'VISITNUM', 'LBDY']


corr = [[]]
for j in df.columns:
    corr[0].append(j)

# correlation
k = 1
for i in num:
    corr.append([])
    for j in df.columns:
        if j != i:
            df1 = df[[i,j]].dropna()
            if j in cat:
                corr[k].append(abs(scipy.stats.kendalltau(df1[i], df1[j])[1]))
            if j in num:
                corr[k].append(abs(scipy.stats.spearmanr(df1[i], df1[j])[1]))
        else:
            corr[k].append(1)
    k += 1


# classes
corr_result = [[], [], [], []]
errors = []
for i in num[:12]:

    #choose varuebles for clusterisation
    v_num = num.index(i)
    var_i = [i]
    for k in range(len(num)):
        if corr[v_num+1][k] < 0.05:
            var_i.append(corr[0][k])
    df_fc = df[var_i].dropna()

    clus_type = -1
    t = 0

    #DBSCAN
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df_fc)
    distances, indices = nbrs.kneighbors(df_fc)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    dis_2d = []
    for l in range(len(distances)):
        dis_2d.append([l, distances[l]])
    rotor = Rotor()
    rotor.fit_rotate(dis_2d)
    elbow_index = rotor.get_elbow_index()
    num_sam = 2*len(var_i)
    metr_d = 0
    clustering = DBSCAN(eps=distances[elbow_index], min_samples=num_sam).fit(df_fc)
    real_result_d = clustering
    for eps in range(-10,10):
        for ms in range(num_sam - int(num_sam/4), num_sam+int(num_sam/4)):
            clustering = DBSCAN(eps=distances[elbow_index]+distances[elbow_index]*eps/100, min_samples=ms).fit(df_fc)
            if len(unique(clustering.labels_)) > 1:
                metr_dif = metrics.silhouette_score(df_fc, clustering.labels_)
                if metr_dif > metr_d:
                    real_result_d = clustering
                    metr_d = metr_dif

    #k-means
    metr_k = 0
    real_result_k=KMeans(n_clusters=2, random_state=0).fit(df_fc)
    for nk in range(2, int(len(var_i)/2)+2):
        clustering = KMeans(n_clusters=nk, random_state=0).fit(df_fc)
        metr_dif = metrics.silhouette_score(df_fc, clustering.labels_)
        if metr_dif > metr_k:
            real_result_k = clustering
            metr_k = metr_dif

    #assign classes for clustering results and check the dependence by the method of correlation analysis
    labels = real_result_d.labels_
    df_fc["LD"] = labels
    labels = real_result_k.labels_
    df_fc["LK"] = labels
    values_clusters_maked = df_fc[[i, 'LD', 'LK']]
    corr_result[0].append(i)

    if metr_d == 0:
        corr_result[1].append(1)
    else:
        values_classes_maked = make_lables(values_clusters_maked, mark='LD', t=2)
        corr_result[1].append(scipy.stats.kendalltau(values_classes_maked[0], values_classes_maked[-1])[1])


    if metr_k == 0:
        corr_result[2].append(1)
    else:
        values_classes_maked = make_lables(values_clusters_maked, mark='LK', t=1)
        corr_result[2].append(scipy.stats.kendalltau(values_classes_maked[0], values_classes_maked[-1])[1])
        if isnan(scipy.stats.kendalltau(values_classes_maked[0], values_classes_maked[-1])[1]):
            print(scipy.stats.kendalltau(values_classes_maked[0], values_classes_maked[-1])[1])
            print(values_classes_maked[0])
            print(values_classes_maked[-1])

    corr_result[3].append(min(corr_result[1][-1], corr_result[2][-1]))

print(corr_result[0])
print(corr_result[1])
print(corr_result[2])
print(corr_result[3])
for i in range(len(corr_result[0])):
    print(corr_result[0][i])
    print(corr_result[1][i])
    print(corr_result[2][i])
    print(corr_result[3][i])

