# Multivariate-Distance-Based-MANOVA - Cited Work and Class Parameters
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
       
# Multivariate-Distance-Based-MANOVA - Fit Method Parameters:
        Input:
        ------
        X: Numpy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc)

        y: Numpy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:
        ------
        self: A fitted model

# Multivariate-Distance-Based-MANOVA - How to Use (Assuming Properly Constructed Input Data):

from welch_manova import WelchMANOVA

stats_analysis = WelchMANOVA().fit(X, y)
