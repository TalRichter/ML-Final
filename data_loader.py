import os

import pandas as pd
import scipy.io
from scipy.io import arff

data = {
    "bioconductor": ["ALL.csv", "ayeastCC.csv", "bcellViper.csv", "bladderbatch.csv", "breastCancerVDX.csv",
                     "CLL.csv", "COPDSexualDimorphism.data.csv", "curatedOvarianData.csv", "DLBCL.csv",
                     "leukemiasEset.csv"],
    "microbiomic": ["BP.csv", "CBH.csv", "CS.csv", "CSS.csv", "FS.csv", "FSH.csv", "PBS.csv", "PDX.csv"],
    "ARFF": ["Breast.arff", "CNS.arff", "Colon.arff", "Lung.arff", "Leukemia.arff",
             "Lymphoma.arff", "MLL.arff", "SRBCT.arff", "Leukemia_3c.arff", "Leukemia_4c.arff"],
    "scikit_feature_datasets": ["ALLAML.mat", "arcene.mat", "BASEHOCK.mat", "Carcinom.mat", "gisette.mat", "Yale.mat",
                                "ORL.mat", "madelon.mat", "lung_small.mat", "Isolet.mat", "GLIOMA.mat"]
}

bioconductor_folder = 'bioconductor'
microbiomic_folder = "microbiomic"
ARFF_folder = 'ARFF'
scikit_folder = 'scikit_feature_datasets'


# loading files from various formats to pandas dataframe, where each class is the last column
# the classes are factorized


def load_bioconductor(base_filename):
    '''
    :param base_filename: files from bioconductor folder
    :return: pandas dataframe
    '''
    dir_name = bioconductor_folder
    path = os.path.join(dir_name, base_filename)
    df = pd.read_csv(path, index_col=0).T
    df = df.rename(columns={df.columns[0]: "Y"})
    temp_cols = df.columns.tolist()
    new_cols = temp_cols[1:] + list(temp_cols[0])
    df = df[new_cols]
    df["Y"] = pd.factorize(df['Y'])[0]
    # xxxsads = {'y': df.iloc[:, -1:], 'X': df.iloc[:, :-1]}
    return df


def load_microbiomic(base_filename):
    dir_name = microbiomic_folder
    path = os.path.join(dir_name, base_filename)
    df = pd.read_csv(path, header=None).T
    df = df[1:]
    df["Y"] = pd.factorize(df.iloc[:,0])[0]
    df = df.iloc[:, 1:]
    df = df.add_prefix('f_')
    return df


def load_ARFF(base_filename):
    '''
    :param base_filename: files from ARFF folder
    :return: pandas dataframe
    '''
    dir_name = ARFF_folder
    path = os.path.join(dir_name, base_filename)
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    df = df.rename(columns={df.columns[-1]: "Y"})
    df["Y"] = pd.factorize(df['Y'])[0]
    return df


def load_scikit_feature_datasets(base_filename):
    '''
    :param base_filename: files from bioconductor folder
    :return: pandas dataframe
    '''
    dir_name = scikit_folder
    path = os.path.join(dir_name, base_filename)
    mat = scipy.io.loadmat(path)
    df = pd.DataFrame(mat['X'], columns=['f_' + str(i) for i in range(mat['X'].shape[1])])
    df['Y'] = mat['Y']
    df["Y"] = pd.factorize(df['Y'])[0]
    return df


def load_toy_example(file):
    return pd.read_csv(file,header=None)
