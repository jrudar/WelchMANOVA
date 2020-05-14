## Disclaimer: Very early release code. In testing stage only. Do not use in production and let me know of any potential issues and fixes with it.
    

## Cited Work and Class Parameters
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

            From Scikit-Learn: "cityblock", "cosine", "euclidean", "l1", "l2",
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
        Deicode package will run for. Only use if metric is 'compositional'.

    comp_method: str, default = "ovo"
        If multiple treatment groups are present, this parameter specifies
        if pairwise comparisons are conducted in a one-vs-one or one-vs-rest
        manner.

        Possible options are:
            "ovo" - One vs. One
            "ovr" - One vs. Rest

    alpha: float, default = 0.05
        Threshold for significance

    method: str, default = "b"
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
       
## Fit Method Parameters:
        Input:
        ------
        X: Numpy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc)

        y: Numpy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:
        ------
        self: A fitted model

## Requirements:
        The latest versions of the following packages:
        Pandas, Scikit-Learn, Scikit-Bio, Deicode, NumPy, and Statsmodels

## How to Use (Assuming Properly Constructed Input Data):

from .welch_manova import WelchMANOVA

stats_analysis = WelchMANOVA().fit(X, y)
