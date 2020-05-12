from sklearn.metrics.pairwise import pairwise_distances

from pandas import DataFrame

from deicode.matrix_completion import MatrixCompletion
from deicode.preprocessing import rclr

from numpy import asarray, power, copy, sqrt
from numpy.random import shuffle, choice

from statsmodels.stats.multitest import multipletests

def manova(D, y, n_perm):

    y_set = asarray(list(set(y)))

    k = y_set.shape[0]

    treatments = []
    for i in range(k):
        tmp = []
        for j in range(y.shape[0]):
            if y[j] == y_set[i]:
                tmp.append(j)

        treatments.append(asarray(tmp))

    n = asarray([treatment.shape[0] for treatment in treatments])

    #Calculate the treatment weights and the sum of squared distances
    W = 0.0
    treatment_w = []
    SSD = []
    for j in range(k):
        treatment_index = treatments[j]

        n_j = n[j] 

        SSD_j = 0.0
        for p in range(0, treatment_index.shape[0] - 1):
            p_index = treatment_index[p]

            for q in range(p+1, treatment_index.shape[0]):
                q_index = treatment_index[q]
                SSD_j += power(D[p_index, q_index], 2)

        #Calculate weight for each treatment
        w_j = (power(n_j, 2) * (n_j - 1)) / SSD_j
        W += w_j
        treatment_w.append(w_j)

        SSD.append(SSD_j)

    #This section calculates the weighted sum squared distances between treatments
    wSSD = 0.0

    for i in range(0, k - 1):
        #Retrieve data for the i-th treatment
        w_i = treatment_w[i]
        n_i = n[i]
        SSD_i = SSD[i]

        wSSD_i = SSD_i / n_i

        for j in range(i+1, k):
            #Retrieve data for the i-th treatment
            w_j = treatment_w[j]
            n_j = n[j]
            SSD_j = SSD[j]

            wSSD_j = SSD_j / n_j
            
            #Calculate all distances for the i-th and j-th treatment
            index_values = asarray([ii for ii, cluster_name in enumerate(y) 
                                    if cluster_name == y_set[i] 
                                    or cluster_name == y_set[j]])

            wSSD_ij = 0.0
            for ii in range(0, index_values.shape[0]):
                i_index = index_values[ii]
                for jj in range(ii+1, index_values.shape[0]):
                    j_index = index_values[jj]

                    wSSD_ij += power(D[i_index, j_index], 2)

            wSSD_ij = wSSD_ij / (n_i + n_j)

            wSSD += w_i * w_j * (((n_i + n_j) * (wSSD_ij - (wSSD_i + wSSD_j))) / (n_i * n_j))

    wSSD = wSSD / W

    #Calculate the sum of all h_j
    h_sum = 0.0
    for j, w_j in enumerate(treatment_w):
        n_j = n[j]

        h_j = w_j / W
        h_j = 1 - h_j
        h_j = h_j / (n_j - 1)
        h_sum += h_j

    #Calculate W*d
    W_num = wSSD / (k - 1)

    W_den_1 = 2 * (k - 2)
    W_den_2 = power(k, 2) - 1
    W_den = 1 + (W_den_1 / W_den_2) * h_sum

    W_star = W_num / W_den

    #Calculated the permuted statistic
    n_greater = 0
    p = None
    if n_perm > 0:
        y_copy = copy(y)
        for i in range(n_perm):
            shuffle(y_copy)

            W_perm = manova(D, y_copy, 0)[0]

            if W_perm >= W_star:
                n_greater += 1

        #The number 1 is added to both the numerator and denominator 
        #since we already have at least 1 possible permutation (W_star)
        p = (float(n_greater) + 1) / (float(n_perm) + 1)

    return W_star, p

def ttest(D, y, A, B, n_perm):

    #Get the index of for every member of group A and group B and
    #permute if necessary
    index_values = asarray([i for i, cluster_name in enumerate(y) 
                            if cluster_name == A 
                            or cluster_name == B])

    A_index = asarray([i for i, cluster_name in enumerate(y) 
                       if cluster_name == A])

    B_index = asarray([i for i, cluster_name in enumerate(y) 
                       if cluster_name == B])

    if n_perm == 0:
        shuffle(index_values)

        A_index = choice(index_values, A_index.shape[0], replace = False)

        B_index = asarray(list(set(index_values) - set(A_index)))

    treatments = [A_index, B_index]
    n = [A_index.shape[0], B_index.shape[0]]

    #Calculate the sum of squared distances and variance of each treatment
    SSD = []
    Var = []
    for j, treatment in enumerate(treatments):
        n_j = n[j] 

        SSD_j = 0.0
        for p in range(0, treatment.shape[0] - 1):
            i_index = treatment[p]
            for q in range(p+1, treatment.shape[0]):
                j_index = treatment[q]
                SSD_j += power(D[i_index, j_index], 2)

        #Calculate variance within each treatment
        var_j = SSD_j / (n_j * (n_j - 1))
        Var.append(var_j)

        #Calculated the weighted SSD
        SSD_j = SSD_j / n_j
        SSD.append(SSD_j)

    #Retrieve data for treatment A
    n_i = n[0]
    wSSD_i = SSD[0]

    #Retrieve data for the j-th treatment
    n_j = n[1]
    wSSD_j = SSD[1]
            
    #Calculate for treatments A and B combined
    wSSD_ij = 0.0
    for ii in range(index_values.shape[0]):
        i_index = index_values[ii]
        for jj in range(ii+1, index_values.shape[0]):
            j_index = index_values[jj]

            wSSD_ij += power(D[i_index, j_index], 2)

    wSSD_ij = wSSD_ij / (n_i + n_j)

    #Calculate T2w
    T2w_num = ((n[0] + n[1]) / (n[0] * n[1])) * (wSSD_ij - wSSD_i - wSSD_j)

    T2w_den = Var[0] + Var[1]

    T2w = T2w_num / T2w_den

    #Calculate the effect size (Cohen's D and R2)
    d2_num = T2w_num

    d2_den = T2w_den / (n_i + n_j - 2)

    d2 = d2_num / d2_den

    d = sqrt(d2)

    #Calculated the permuted statistic
    n_greater = 0
    p = None
    if n_perm > 0:
        for i in range(n_perm):
            T_perm = ttest(D, y, A, B, 0)[0]

            if T_perm >= T2w:
                n_greater += 1

        #The number 1 is added to both the numerator and denominator 
        #since we already have at least 1 possible permutation (W_star)
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

    n_perm: int, default = 999
        The number of permutations

    is_compositional: bool, default = True
        Whether to construct a robust Aitchison distance matrix.

    alpha: float, default = 0.05
        Threshold for significance

    method: str, default = "b"
        Method which is used to adjust p-values if more than two comparisons
        are made.

    """
    def __init__(self, 
                 metric = "euclidean",
                 n_perm = 999,
                 is_compositional = False,
                 n_comp = 2,
                 max_iter = 10,
                 alpha = 0.05,
                 method = "b"):

        self.metric = metric
        self.n_perm = n_perm
        self.is_compositional = is_compositional
        self.n_comp = n_comp
        self.max_iter = max_iter
        self.alpha = alpha
        self.method = method

    def fit(self, X, y):
        """
        Input:
        ------
        X: Numpy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc)

        y: Numpy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:
        ------
        self: A fitted model
        """

        #Transform into a distance matrix
        if self.is_compositional == False:
            D = pairwise_distances(X, 
                                   metric = self.metric)

        elif self.is_compositional == True:
            X_trf = rclr(DataFrame(X).astype(float))

            D = MatrixCompletion(self.n_comp, 
                                 self.max_iter).fit(X_trf).distance

        #Save data
        self.D = D
        self.X = X
        self.y = y

        #Calculate permuted statistic
        if len(set(self.y)) > 2:
            W_star, p_value = manova(self.D, 
                                     self.y,
                                     self.n_perm)

            #Save data
            self.W_star_d = W_star
            self.p_value = p_value

            if self.p_value >= self.alpha:
                print ("Test Statisitic: ", self.W_star_d)
                print ("p-value: ", self.p_value)

            else:
                print ("Test Statisitic: ", self.W_star_d)
                print ("p-value: ", self.p_value, "\n")

                self._pairwise()

        else:
            self._pairwise()

        return self

    def _pairwise(self):
        
        #Calculate p-values
        y_set = list(set(self.y))

        self.comparision = []
        self.test_statistics = []
        self.raw_p_values = []
        self.effect_size = []
        for i in range(0, len(y_set) - 1):
            for j in range(i+1, len(y_set)):
                raw_pairwise = ttest(self.D, 
                                        self.y,
                                        y_set[i],
                                        y_set[j],
                                        self.n_perm)

                self.comparision.append("%s-%s" %(y_set[i], y_set[j]))
                self.test_statistics.append(raw_pairwise[0])
                self.raw_p_values.append(raw_pairwise[1])
                self.effect_size.append(raw_pairwise[2])

        #Adjust for multiple comparisons
        if len(y_set) > 2:
            self.adjusted_p_values = multipletests(self.raw_p_values,
                                                    self.alpha,
                                                    self.method)

            self.table_result = DataFrame(data = [self.test_statistics,
                                                    self.raw_p_values,
                                                    self.adjusted_p_values,
                                                    self.effect_size],
                                            index = ["T2_d Statistic",
                                                    "Raw p-value",
                                                    "Adjusted p-value",
                                                    "Effect Size (Cohen's d)"],
                                            columns = self.comparision).transpose()

        else:
            self.table_result = DataFrame(data = [self.test_statistics,
                                                    self.raw_p_values,
                                                    self.effect_size],
                                            index = ["T2_d Statistic",
                                                    "Raw p-value",
                                                    "Effect Size (Cohen's d)"],
                                            columns = self.comparision).transpose()

        print(self.table_result)

