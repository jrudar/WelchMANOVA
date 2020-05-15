from sklearn.metrics.pairwise import pairwise_distances

from pandas import DataFrame

from skbio.stats.distance import DissimilarityMatrix

from deicode.matrix_completion import MatrixCompletion
from deicode.preprocessing import rclr

from numpy import asarray, power, copy, sqrt, float32, hstack
from numpy.random import shuffle

from statsmodels.stats.multitest import multipletests

def manova(D, y, n_perm):

    """
    Get the index of for every member of group A and group B and
    permute if necessary
    """
    y_set = asarray(list(set(y)))

    treatments = []
    if n_perm == 0:
        y_copy = copy(y)
        shuffle(y_copy)

        index_values = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y_copy)])

        for treatment in y_set:
            treatments.append(asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y_copy) 
                                       if cluster_name == treatment]))

    elif n_perm > 0:
        index_values = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y)])

        for treatment in y_set:
            treatments.append(asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y) 
                                       if cluster_name == treatment]))

    D_obj = DissimilarityMatrix(D, index_values)

    #Calculate k
    k = y_set.shape[0]

    n = asarray([treatment.shape[0] for treatment in treatments])

    #Implementaion of Equations 4, 6, and 7
    eqn_4 = 0.0
    w_treatments = []
    SSDs = []
    for i in range(k):
        treatment_ids = treatments[i]

        n_i = n[i] 

        #Sum squared distances within treatment groups
        SSD_i = (D_obj.within(treatment_ids).values[:,2].sum() / 2.0)
        SSDs.append(SSD_i)

        #Equation 6 (Calculation of Within Treatment Variance)
        eqn_6 = SSD_i / (n_i * (n_i - 1))

        #Equation 7 (Calculation of w_j)
        eqn_7 = n_i / eqn_6
        w_treatments.append(eqn_7)

        #Equation 4 (Sum of w_js)
        eqn_4 += eqn_7

    #Implementation of equations 16 and 17
    eqn_16_num = 0.0
    for i in range(0, k - 1):
        treatment_i = treatments[i]

        #Retrieve data for the i-th treatment
        w_i = w_treatments[i]
        n_i = n[i]
        SSD_i = SSDs[i]

        #Section of Equation 17 for Treatment i
        wSSD_i = SSD_i / n_i

        for j in range(i+1, k):
            treatment_j = treatments[j]

            #Retrieve data for the j-th treatment
            w_j = w_treatments[j]
            n_j = n[j]
            SSD_j = SSDs[j]

            #Section of Equation 17 for Treatment j
            wSSD_j = SSD_j / n_j
            
            #Section of Equation 17 for Treatments i and j
            SSD_ij = (D_obj.filter(hstack((treatment_i, treatment_j))).data.sum() / 2.0)

            wSSD_ij = SSD_ij / (n_i + n_j)

            #Summation for Equations 16 using Equation 17
            eqn_16_num += w_i * w_j * (((n_i + n_j) * (wSSD_ij - (wSSD_i + wSSD_j))) / (n_i * n_j))

    #Equantion 16
    eqn_16 = eqn_16_num / eqn_4

    #Impementation of Equation 5
    sum_h = 0.0
    for i, w_i in enumerate(w_treatments):
        n_i = n[i]

        eqn_5_num = power(((1-w_i) / eqn_4), 2)
        
        eqn_5 = eqn_5_num / (n_i - 1)

        sum_h += eqn_5

    #Calculate W*d
    W_num = eqn_16 / (k - 1)

    W_den_1 = 2 * (k - 2)
    W_den_2 = power(k, 2) - 1
    W_den = 1 + (W_den_1 / W_den_2) * sum_h

    W_star = W_num / W_den

    #Calculated the permuted statistic
    n_greater = 0
    p = None
    if n_perm > 0:
        for i in range(n_perm):
            W_perm = manova(D, y, 0)[0]

            if W_perm >= W_star:
                n_greater += 1

        #The number 1 is added to both the numerator and denominator 
        #since we already have at least 1 possible permutation (W_star)
        p = (float(n_greater) + 1) / (float(n_perm) + 1)

    return W_star, p

def ttest_ovr(D, y, A, n_perm):

    """
    Get the index of for every member of group A and group B and
    permute if necessary
    """
    index_values = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y)])

    A_index = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y) 
                       if cluster_name == A])

    B_index = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y) 
                       if cluster_name != A])

    if n_perm == 0: shuffle(index_values)

    D_obj = DissimilarityMatrix(D, index_values)

    #Calculate the within-group and between-group differences
    SSD_i = (D_obj.within(A_index).values[:,2].sum() / 2.0)
    SSD_j = (D_obj.within(B_index).values[:,2].sum() / 2.0)
    SSD_ij = (D_obj.data.sum() / 2.0)

    #Calculate the sample sizes
    n_i = A_index.shape[0]
    n_j = B_index.shape[0]

    #Calculate the weighted sum of squared distances and variance
    wSSD_i = SSD_i / n_i
    wSSD_j = SSD_j / n_j
    wSSD_ij = SSD_ij / (n_i + n_j)

    var_i = SSD_i / (n_i * (n_i - 1))
    var_j = SSD_j / (n_j * (n_j - 1))

    #Calculate T2w
    T2w_num = ((n_i + n_j) / (n_i * n_j)) * (wSSD_ij - wSSD_i - wSSD_j)

    T2w_den = var_i + var_j

    T2w = T2w_num / T2w_den

    #Calculate the effect size (Cohen's D)
    if n_perm == 0:
        d = None

    elif n_perm > 0:
        d2_num = T2w_num

        d2_den = T2w_den / (n_i + n_j - 2)

        d2 = d2_num / d2_den

        d = sqrt(d2)

    #Calculated the permuted statistic
    n_greater = 0
    p = None
    if n_perm > 0:
        for i in range(n_perm):
            T_perm = ttest_ovr(D, y, A, 0)[0]

            if T_perm >= T2w:
                n_greater += 1

        #The number 1 is added to both the numerator and denominator 
        #since we already have at least 1 possible permutation (T2w)
        p = (float(n_greater) + 1) / (float(n_perm) + 1)

    return T2w, p, d

def ttest_ovo(D, y, A, B, n_perm):

    """
    Get the index of for every member of group A and group B and
    permute if necessary
    """
    index_values = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y)])

    A_index = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y) 
                       if cluster_name == A])

    B_index = asarray(["Sample %s" %str(i) for i, cluster_name in enumerate(y) 
                       if cluster_name == B])

    D_obj = DissimilarityMatrix(D, index_values).filter(hstack((A_index, B_index)))

    if n_perm == 0:

        D_ss = D_obj.data

        index_values = asarray(D_obj.ids)

        shuffle(index_values)

        D_obj = DissimilarityMatrix(D_ss, index_values)

    #Calculate the within-group and between-group differences
    SSD_i = (D_obj.within(A_index).values[:,2].sum() / 2.0)
    SSD_j = (D_obj.within(B_index).values[:,2].sum() / 2.0)
    SSD_ij = (D_obj.data.sum() / 2.0)

    #Calculate the sample sizes
    n_i = A_index.shape[0]
    n_j = B_index.shape[0]

    #Calculate the weighted sum of squared distances and variance
    wSSD_i = SSD_i / n_i
    wSSD_j = SSD_j / n_j
    wSSD_ij = SSD_ij / (n_i + n_j)

    var_i = SSD_i / (n_i * (n_i - 1))
    var_j = SSD_j / (n_j * (n_j - 1))

    #Calculate T2w
    T2w_num = ((n_i + n_j) / (n_i * n_j)) * (wSSD_ij - wSSD_i - wSSD_j)

    T2w_den = var_i + var_j

    T2w = T2w_num / T2w_den

    #Calculate the effect size (Cohen's D)
    if n_perm == 0:
        d = None

    elif n_perm > 0:
        d2_num = T2w_num

        d2_den = T2w_den / (n_i + n_j - 2)

        d2 = d2_num / d2_den

        d = sqrt(d2)

    #Calculated the permuted statistic
    n_greater = 0
    p = None
    if n_perm > 0:
        for i in range(n_perm):
            T_perm = ttest_ovo(D, y, A, B, 0)[0]

            if T_perm >= T2w:
                n_greater += 1

        #The number 1 is added to both the numerator and denominator 
        #since we already have at least 1 possible permutation (T2w)
        p = (float(n_greater) + 1) / (float(n_perm) + 1)

    return T2w, p, d

class WelchMANOVA():
    """
    This code is based off of the following work:

    Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A, Knight R, and
    Zengler K. (2019). A Novel Sparse Compositional Technique Reveals Microbial 
    Perturbations. mSystems. 4(1): e00016-19. doi: 10.1128/mSystems.00016-19

    Alekseyenko AV. (2016). Multivariate Welch t-test on distances. Bioinformatics.
    32(23): 3552-3558. doi: 10.1093/bioinformatics/btw524

    Hamidi B, Wallace K, Vasu C, and Alekseyenko AV. (2019). W∗d -test: robust 
    distance-based multivariate analysis of variance. Microbiome. 7(1): 51.
    doi: doi: 10.1186/s40168-019-0659-9.

    Input:
    ------
    metric: str, default = "euclidean"
        The metric used to construct comparisons between samples.

        The possible options are:
            "compositional" - Constructs a distance matrix using Deicode

            From Scikit-Learn - "cityblock", "cosine", "euclidean", "l1", "l2",
            "manhattan", "braycurtis", "canberra", "chebyshev", "correlation",
            "dice", "hamming", "jaccard", "kulsinski", "rogerstanimoto", 
            "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
            "sqeuclidean", "yule"

            "precomputed" - User supplied square distance matrix

    n_perm: int, default = 999
        The number of permutations

    n_comp: int, default = 2
        The number of components the matrix completion algorithm of the
        Deicode package will extract. Only used if metric is 'compositional'.

    max_iter: int, default = 10
        The number of iterations the matrix completion algorithm of the
        Deicode package will run for. Only used if metric is 'compositional'.

    comp_method: str, default = "ovo"
        If multiple treatment groups are present, this parameter specifies
        if pairwise comparisons are conducted in a one-vs-one or one-vs-rest
        manner.

        Possible options are:
            "ovo" - One vs. One
            "ovr" - One vs. Rest

    alpha: float, default = 0.05
        Threshold for significance for conducting pairwise tests.

    method: str, default = "fdr_bh"
        Method which is used to adjust p-values if more than two comparisons
        are made. Possible options are:
            "b" - Bonferroni
            "s" - Sidak
            "h" - Holm
            "hs" - Holm-Sidak
            "sh" - Simes-Hochberg
            "ho" - Hommel
            "fdr_bh" - FDR Benjamini-Hochberg
            "fdr_by" - FDR Benjamini-Yekutieli
            "fdr_tsbh" - FDR Two-Stage Benjamini-Hochberg
            "fdr_tsbky" - FDR Two-Stage Benjamini-Krieger-Yekutieli
            "fdr_gbs" - FDR Adaptive Gavrilov-Benjamini-Sarkar

    """
    def __init__(self, 
                 metric = "euclidean",
                 n_perm = 999,
                 n_comp = 2,
                 max_iter = 10,
                 comp_method = "ovo",
                 alpha = 0.05,
                 method = "fdr_bh"):

        self.metric = metric
        self.n_perm = n_perm
        self.n_comp = n_comp
        self.max_iter = max_iter
        self.comp_method = comp_method
        self.alpha = alpha
        self.method = method

    def fit(self, X, y):
        """
        Input:
        ------
        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc). If X is a distance matrix,
        it should be of shape (m, m).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:
        ------
        self: A fitted model
        """

        #Transform into a distance matrix
        if self.metric != "compositional":
            if self.metric != "precomputed":
                D = pairwise_distances(X, 
                                       metric = self.metric).astype(float32)

            else:
                D = X

        elif self.metric == "compositional":
            X_trf = rclr(DataFrame(X).astype(float))

            D = MatrixCompletion(self.n_comp, 
                                 self.max_iter).fit(X_trf).distance.astype(float32)

        #Save data
        self.D = D
        self.X = X
        self.y = y

        #Calculate permuted statistic
        W_star, p_value = manova(power(self.D, 2), 
                                 self.y,
                                 self.n_perm)

        #Save data
        self.Wd_star = W_star
        self.p_value = p_value

        print ("Wd*: ", self.Wd_star)
        print ("p-value: ", self.p_value)

        if len(set(self.y)) > 2 and self.p_value <= self.alpha: self._pairwise()

        return self

    def _pairwise(self):
        
        y_set = list(set(self.y))

        self.comparision = []
        self.test_statistics = []
        self.raw_p_values = []
        self.effect_size = []

        if self.comp_method == "ovo":
            for i in range(0, len(y_set) - 1):
                for j in range(i+1, len(y_set)):
                    raw_pairwise = ttest_ovo(power(self.D, 2), 
                                             self.y,
                                             y_set[i],
                                             y_set[j],
                                             self.n_perm)

                    self.comparision.append("%s-%s" %(y_set[i], y_set[j]))
                    self.test_statistics.append(raw_pairwise[0])
                    self.raw_p_values.append(raw_pairwise[1])
                    self.effect_size.append(raw_pairwise[2])

        elif self.comp_method == "ovr":
            for i in range(0, len(y_set)):
                raw_pairwise = ttest_ovr(power(self.D, 2), 
                                         self.y,
                                         y_set[i],
                                         self.n_perm)

                self.comparision.append("%s-Rest" %y_set[i])
                self.test_statistics.append(raw_pairwise[0])
                self.raw_p_values.append(raw_pairwise[1])
                self.effect_size.append(raw_pairwise[2])

        #Adjust for multiple comparisons
        if len(y_set) > 2:
            self.adjusted_p_values = multipletests(self.raw_p_values,
                                                   self.alpha,
                                                   self.method)[1]

            self.table_result = DataFrame(data = [self.test_statistics,
                                                  self.raw_p_values,
                                                  self.adjusted_p_values,
                                                  self.effect_size],
                                          index = ["T2w Statistic",
                                                   "Raw p-value",
                                                   "Adjusted p-value",
                                                   "Effect Size (Cohen's d)"],
                                          columns = self.comparision).transpose()

        else:
            self.table_result = DataFrame(data = [self.test_statistics,
                                                  self.raw_p_values,
                                                  self.effect_size],
                                          index = ["T2w Statistic",
                                                   "p-value",
                                                   "Effect Size (Cohen's d)"],
                                          columns = self.comparision).transpose()

        print(self.table_result)

