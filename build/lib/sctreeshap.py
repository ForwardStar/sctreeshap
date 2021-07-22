__name__ = 'sctreeshap'
__version__ = "0.5.0rc1"

import time
import threading
import numpy as np
import anndata as ad
import pandas as pd
import shap
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class sctreeshap:
    def __checkLoops(self, root):
        checkResult = True
        self.__visited.append(root)
        if root not in self.__TreeNode:
            return True
        for item in self.__TreeNode[root]:
            if item in self.__visited:
                return False
            else:
                checkResult = self.__checkLoops(item)
        return checkResult

    def __checkClusterTree(self, tree_arr):
        if not isinstance(tree_arr, dict):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') requires a dict object as input.")
            return -1
        typeOfTree = None
        for key in tree_arr.keys():
            if not isinstance(tree_arr[key], list) and not isinstance(tree_arr[key], tuple):
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (wrong format).")
                return False
            if typeOfTree is None:
                if len(tree_arr[key]) > 1 and isinstance(tree_arr[key][1], int):
                    typeOfTree = "ParentPointer"
                else:
                    typeOfTree = "ChildPointer"
            else:
                if len(tree_arr[key]) > 1 and isinstance(tree_arr[key][1], int):
                    if typeOfTree == "ChildPointer":
                        print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (wrong format).")
                        return False
                else:
                    if typeOfTree == "ParentPointer":
                        print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (wrong format).")
                        return False
        if typeOfTree == "ChildPointer":
            self.__TreeNode = tree_arr
            for key in tree_arr.keys():
                for item in tree_arr[key]:
                    if item not in self.__parent.keys():
                        self.__parent[item] = key
                    else:
                        print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (not a tree structure).")
                        return False
            for key in tree_arr.keys():
                if key not in self.__parent.keys():
                    if self.__root is None:
                        self.__root = key
                    else:
                        print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (not a tree structure).")
                        return False
            if self.__root is None:
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (not a tree structure).")
                return False
            if not self.__checkLoops(self.__root):
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (not a tree structure).")
                return False
        else:
            # needs implementation
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives a valid but not yet supported format. Please contact the developer.")
            return False
        return True

    # Construct a sctreeshap object with a given cluster tree.
    # tree_arr: dictionary, can be in 2 different formats:
    #           1. Let n <- len(tree_arr);
    #                   then n represents the number of non-leaf nodes in the cluster tree;
    #               tree_arr[str] represents the node of name str in the cluster tree;
    #                   tree_arr[str] can be a list or a tuple of strings, representing the name of childs of the node (from left to right);
    #                   e.g. tree_arr['n1'] = ('n2', 'n70') represents a node named 'n1', whose left child is 'n2' and right child is 'n70';
    #               note that you do not need to create nodes for clusters, since they are leaf nodes and have no childs.
    #            2. Let n <- len(tree_arr);
    #                   then n represents the number of nodes (root excluded) in the cluster tree;
    #               tree_arr[str] represents the node of name str in the cluster tree;
    #                   tree_arr[str] should be a list or a tuple of a string and an int, representing the name of parent of the node and which child it is (from left to right, start from 0);
    #                   e.g. tree_arr['n2'] = ('n1', 0) represents a node named 'n2', who is the leftmost child of 'n1';
    #               note that you do not need to create a node for the root, since it does not have a parent.
    # If tree_arr is None, then it does not construct a cluster tree.
    def __init__(self, tree_arr=None):
        self.numOfClusters = 0
        self.__clusterDict = {}
        self.__dataDirectory = None
        self.__dataSet = None
        self.__branch = None
        self.__cluster = None
        self.__clusterSet = []
        self.__waitingMessage = None
        self.__isFinished = False
        self.__XGBClassifer = None
        self.__explainer = None
        self.__shapValues = None
        self.__maxDisplay = None
        self.__topGenes = None
        self.__featureNames = None
        self.__TreeNode = None
        self.__parent = {}
        self.__root = None
        self.__visited = []
        self.__shapParamsBinary = {
            "max_display": 10,
            "bar_plot": True,
            "beeswarm": True,
            "force_plot": False,
            "heat_map": False,
            "decision_plot": False
        }
        self.__shapParamsMulti = {
            "max_display": 10,
            "bar_plot": True,
            "beeswarm": False,
            "decision_plot": False
        }
        if tree_arr is None:
            return None
        if not self.__checkClusterTree(tree_arr):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') throws an exception.")
            return -1

    # Set default data directory.
    # data_directory: a string representing the directory of the default input file.
    def setDataDirectory(self, data_directory=None):
        if data_directory is None:
            self.__dataDirectory = None
            return None
        if not isinstance(data_directory, str):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setDataDirectory()' (in file '" + __file__ + "') receives an invalid data_directory of wrong type.")
            return -1
        self.__dataDirectory = data_directory
        return None
    
    # Set default dataset.
    # data: DataFrame or AnnData.
    def setDataSet(self, data=None):
        if data is None:
            self.__dataSet = None
            return None
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setDataSet()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        self.__dataSet = data
        return None
    
    # Set default feature names.
    # feature_names: ndarray, list or tuple.
    def setFeatureNames(self, feature_names=None):
        if feature_names is None:
            self.__featureNames = None
            return None
        if not isinstance(feature_names, np.ndarray) and not isinstance(feature_names, list) and not isinstance(feature_names, tuple):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setFeatureNames()' (in file '" + __file__ + "') receives an invalid feature_names dataset of wrong type (must be 'ndarray', 'list' or 'tuple').")
            return -1
        self.__featureNames = np.array(feature_names)
        return None
    
    # Set default branch.
    # branch_name: str, representing the branch's name, which would be defaultedly chosen.
    def setBranch(self, branch_name=None):
        if branch_name is None:
            self.__branch = None
            return None
        if not isinstance(branch_name, str):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setBranch()' (in file '" + __file__ + "') receives an invalid branch_name of wrong type.")
            return -1
        self.__branch = branch_name
        return None
    
    # Set default cluster.
    # cluster_name: str, representing the cluster's name, which would be defaultedly chosen.
    def setCluster(self, cluster_name=None):
        if cluster_name is None:
            self.__cluster = None
            return None
        if not isinstance(cluster_name, str):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setCluster()' (in file '" + __file__ + "') receives an invalid cluster_name of wrong type.")
            return -1
        self.__cluster = cluster_name
        return None
    
    # Set default target cluster set.
    # cluster_set: a list or tuple of strings containing all target clusters to choose.
    def setClusterSet(self, cluster_set=None):
        if cluster_set is None:
            self.__clusterSet = None
            return None
        if not isinstance(cluster_set, list) and not isinstance(cluster_set, tuple):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setClusterSet()' (in file '" + __file__ + "') receives an invalid cluster_set of wrong type.")
            return -1
        self.__clusterSet = cluster_set
        return None
    
    # Set default shap plots parameters of explainBinary().
    # shap_params: dictionary, including five keys: ["max_display", "bar_plot", "beeswarm", "force_plot", "heat_map", "decision_plot"], which is defaultedly set as [10, True, True, False, False, False];
    #           you can reset the dict to determine what kinds of figures to output, and maximum number of genes you want to depict.
    def setShapParamsBinary(self, shap_params=None):
        if shap_params is None:
            self.__shapParamsBinary = {
                "max_display": 10,
                "bar_plot": True,
                "beeswarm": True,
                "force_plot": False,
                "heat_map": False,
                "decision_plot": False
            }
            return None
        if not isinstance(shap_params, dict):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setShapParamsBinary()' (in file '" + __file__ + "') receives an invalid shap_params of wrong type.")
            return -1
        self.__shapParamsBinary = shap_params
        return None
    
    # Set default shap plots parameters of explainMulti().
    # shap_params: dictionary, including three keys: ["max_display", "bar_plot", "beeswarm", "decision_plot"], which is defaultedly set as [10, True, False, False];
    #           you can reset the dict to determine what kinds of figures to output, and maximum number of genes you want to depict.
    def setShapParamsMulti(self, shap_params=None):
        if shap_params is None:
            self.__shapParamsMulti = {
                "max_display": 10,
                "bar_plot": True,
                "beeswarm": False,
                "decision_plot": False
            }
            return None
        if not isinstance(shap_params, dict):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setShapParamsMulti()' (in file '" + __file__ + "') receives an invalid shap_params of wrong type.")
            return -1
        self.__shapParamsMulti = shap_params
        return None

    # Get XGBClassifier of the last job (available after 'a.explainBinary()' or 'a.explainMulti()').
    # Return: <class 'xgboost.sklearn.XGBClassifier'> object
    def getClassifier(self):
        return self.__XGBClassifer

    # Get shap explainer of the last job (available after 'a.explainBinary()' or 'a.explainMulti()').
    # Return: <class 'shap.explainers._tree.Tree'> object
    def getExplainer(self):
        return self.__explainer
    
    # Get shap values of the last job (available after 'a.explainBinary()' or 'a.explainMulti()').
    # Return: ndarray.
    def getShapValues(self):
        return self.__shapValues
    
    # Get top genes of max absolute mean shap values.
    # max_display: int, the number of top genes you want to derive.
    # shap_values: list or ndarray.
    # feature_names: ndarray, list or tuple.
    # Return: ndarray.
    def getTopGenes(self, max_display=None, shap_values=None, feature_names=None):
        if shap_values is None:
            shap_values = self.__shapValues
        if feature_names is None:
            feature_names = self.__featureNames
        if not isinstance(feature_names, np.ndarray) and not isinstance(feature_names, list) and not isinstance(feature_names, tuple):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.getTopGenes()' (in file '" + __file__ + "') receives an invalid feature_names dataset of wrong type (must be 'ndarray', 'list' or 'tuple').")
            return -1
        if not isinstance(max_display, int):
            max_display = self.__maxDisplay
        if isinstance(shap_values, list):
            # Multi-classification
            if shap_values[0].shape[1] != len(feature_names):
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.getTopGenes()' (in file '" + __file__ + "') found unmatched shap_values and feature_names!")
                return -1
            max_display = min(max_display, len(feature_names))
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):][::-1]
            feature_names = [feature_names[i] for i in feature_order]
            return np.array(feature_names)
        elif isinstance(shap_values, np.ndarray):
            # Binary classification
            if shap_values.shape[1] != len(feature_names):
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.getTopGenes()' (in file '" + __file__ + "') found unmatched shap_values and feature_names!")
                return -1
            max_display = min(max_display, len(feature_names))
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):][::-1]
            feature_names = [feature_names[i] for i in feature_order]
            return np.array(feature_names)
        else:
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.getTopGenes()' (in file '" + __file__ + "') cannot recognize the format of shap_values.")
            return -1

    # Find which branch a given cluster is in.
    # cluster_name: str, representing the cluster's name, e.g. "Exc L5-6 THEMIS FGF10".
    #           if cluster_name is None: choose default cluster.
    # Return: str, representing the path.
    def findCluster(self, cluster_name=None, root=None, path="ROOT"):
        if root is None:
            root = self.__root
        if root is None:
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.findCluster()' (in file '" + __file__ + "') found an empty cluster tree!")
            return -1
        if cluster_name is None:
            cluster_name = self.__cluster
            if cluster_name is None:
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.findCluster()' (in file '" + __file__ + "') requires a target cluster name.")
                return -1
        if root not in self.__TreeNode.keys():
            return "Cluster " + cluster_name + " not found!"
        childs = self.__TreeNode[root]
        for item in childs:
            if item == cluster_name:
                return path + " --> " + root + " --> " + cluster_name
            else:
                result = self.findCluster(cluster_name, item, path + " --> " + root)
                if result != "Cluster " + cluster_name + " not found!":
                    return result
        return "Cluster " + cluster_name + " not found!"
    
    # List the clusters of a given branch.
    # branch_name: str, representing the branch's name, e.g. "n48".
    #           if branch_name is None: choose default branch; if default is still None, list all clusters.
    # Return: list, including all cluster names under the branch.
    def listBranch(self, branch_name=None):
        if self.__root is None:
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.listBranch()' (in file '" + __file__ + "') found an empty cluster tree!")
            return -1
        if branch_name is None:
            branch_name = self.__branch
        try:
            if branch_name is None:
                root = self.__root
            else:
                root = self.__TreeNode[branch_name]
        except:
            return [branch_name]
        result = []
        for item in root:
            result = result + self.listBranch(item)
        return result
    
    # Load default dataset and build default cluster tree.
    # Return: AnnData, the default dataset.
    def loadDefault(self):
        def showProcess():
            print(self.__waitingMessage, end="  ")
            while not self.__isFinished:
                print('\b-', end='')
                time.sleep(0.05)
                print('\b\\', end='')
                time.sleep(0.05)
                print('\b|', end='')
                time.sleep(0.05)
                print('\b/', end='')
                time.sleep(0.05)
            print('\bdone')
        import sys, os
        data_directory = __file__[:-13] + "sctreeshap-" + __version__ + ".dist-info/"
        if not os.path.isfile(data_directory + "INPUT_DATA.h5ad"):
            print("First time loading. Downloading the partitions of dataset... (will need 427.6 MiB)")
            from tqdm import tqdm
            from urllib.request import urlopen, Request
            from pathlib import Path
            import tarfile
            import bz2
            if not os.path.exists(data_directory + "tmp"):
                os.mkdir(data_directory + "tmp")
            for i in range(1, 43):
                part_num = str(i)
                part_num = (3 - len(part_num)) * '0' + part_num
                if os.path.isfile(data_directory + "tmp/INPUT_DATA.h5ad.tar.bz2.part" + part_num):
                    print("Part " + part_num + " has been downloaded. Skipped.")
                    continue
                else:
                    print("Downloading part " + part_num)
                path = Path(data_directory + "tmp/INPUT_DATA.h5ad.tar.bz2.part" + part_num)
                url = "https://raw.githubusercontent.com/ForwardStar/sctreeshap/main/datasets/INPUT_DATA.h5ad.tar.bz2.part" + part_num
                headers = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}
                try:
                    blocksize = 1024 * 8
                    blocknum = 0
                    while True:
                        try:
                            with urlopen(Request(url, headers=headers), timeout=5) as resp:
                                total = resp.info().get("content-length", None)
                                with tqdm(
                                    unit="B",
                                    unit_scale=True,
                                    miniters=1,
                                    unit_divisor=1024,
                                    total=total if total is None else int(total),
                                ) as t, path.open("wb") as f:
                                    block = resp.read(blocksize)
                                    while block:
                                        f.write(block)
                                        blocknum += 1
                                        t.update(len(block))
                                        block = resp.read(blocksize)
                            break
                        except:
                            print("Timed out, retrying...")
                except (KeyboardInterrupt, Exception):
                    # Make sure file doesn’t exist half-downloaded
                    if path.is_file():
                        path.unlink()
                    raise
            
            self.__waitingMessage = "Merging the partitions of dataset..."
            self.__isFinished = False
            thread_merge = threading.Thread(target=showProcess)
            thread_merge.start()
            path = Path(data_directory + "tmp/")
            outfile = open(os.path.join(data_directory, "INPUT_DATA.h5ad.tar.bz2"), 'wb')
            files = os.listdir(path)
            files.sort()
            for file in files:
                filepath = os.path.join(path, file)
                infile = open(filepath, 'rb')
                data = infile.read()
                outfile.write(data)
                infile.close()
            outfile.close()
            self.__isFinished = True
            thread_merge.join()
            time.sleep(0.2)

            self.__waitingMessage = "Extracting the dataset... (will need 5.9 GiB)"
            self.__isFinished = False
            thread_extract = threading.Thread(target=showProcess)
            thread_extract.start()
            archive = tarfile.open(os.path.join(data_directory, "INPUT_DATA.h5ad.tar.bz2"), "r:bz2")
            archive.extractall(data_directory)
            archive.close()
            self.__isFinished = True
            thread_extract.join()
            time.sleep(0.2)

        self.__waitingMessage = "Building default cluster tree..."
        self.__isFinished = False
        thread_build = threading.Thread(target=showProcess)
        thread_build.start()
        tree_arr = {
            "n1": ('n2', 'n70'),
            "n2": ('n26', 'n3'),
            "n3": ('n4', 'n21'),
            "n4": ('n7', 'n5'),
            "n5": ('Exc L5-6 THEMIS DCSTAMP', 'n6'),
            "n6": ('Exc L5-6 THEMIS CRABP1', 'Exc L5-6 THEMIS FGF10'),
            "n7": ('n8', 'Exc L4-5 FEZF2 SCN4B'),
            "n8": ('n9', 'n12'),
            "n9": ('n10', 'Exc L5-6 THEMIS C1QL3'),
            "n10": ('n11', 'Exc L2-3 LINC00507 FREM3'),
            "n11": ('Exc L2 LAMP5 LTK', 'Exc L2-4 LINC00507 GLP2R'),
            "n12": ('n13', 'n17'),
            "n13": ('Exc L3-4 RORB CARM1P1', 'n14'),
            "n14": ('Exc L3-5 RORB ESR1', 'n15'),
            "n15": ('Exc L3-5 RORB COL22A1', 'n16'),
            "n16": ('Exc L3-5 RORB FILIP1L', 'Exc L3-5 RORB TWIST2'),
            "n17": ('n19', 'n18'),
            "n18": ('Exc L5-6 RORB TTC12', 'Exc L4-6 RORB C1R'),
            "n19": ('Exc L4-5 RORB FOLH1B', 'n20'),
            "n20": ('Exc L4-6 RORB SEMA3E', 'Exc L4-5 RORB DAPK2'),
            "n21": ('Exc L4-6 FEZF2 IL26', 'n22'),
            "n22": ('Exc L5-6 FEZF2 ABO', 'n23'),
            "n23": ('n24', 'Exc L5-6 FEZF2 EFTUD1P1'),
            "n24": ('n25', 'Exc L6 FEZF2 OR2T8'),
            "n25": ('Exc L6 FEZF2 SCUBE1', 'Exc L5-6 SLC17A7 IL15'),
            "n26": ('n27', 'n53'),
            "n27": ('n48', 'n28'),
            "n28": ('n41', 'n29'),
            "n29": ('n37', 'n30'),
            "n30": ('n31', 'n34'),
            "n31": ('n32', 'Inh L1-3 VIP GGH'),
            "n32": ('n33', 'Inh L1-3 VIP CCDC184'),
            "n33": ('Inh L1-3 VIP CHRM2', 'Inh L2-4 VIP CBLN1'),
            "n34": ('n36', 'n35'),
            "n35": ('Inh L2-4 VIP SPAG17', 'Inh L1-4 VIP OPRM1'),
            "n36": ('Inh L1-2 VIP LBH', 'Inh L2-3 VIP CASC6'),
            "n37": ('n39', 'n38'),
            "n38": ('Inh L2-5 VIP SERPINF1', 'Inh L2-5 VIP TYR'),
            "n39": ('n40', 'Inh L1-2 VIP PCDH20'),
            "n40": ('Inh L2-6 VIP QPCT', 'Inh L3-6 VIP HS3ST3A1'),
            "n41": ('n43', 'n42'),
            "n42": ('Inh L1-3 VIP ADAMTSL1', 'Inh L1-4 VIP PENK'),
            "n43": ('n44', 'n46'),
            "n44": ('n45', 'Inh L1-2 SST BAGE2'),
            "n45": ('Inh L1 SST CHRNA4', 'Inh L1−2 GAD1 MC4R'),
            "n46": ('Inh L1-3 PAX6 SYT6', 'n47'),
            "n47": ('Inh L1-2 VIP TSPAN12', 'Inh L1-4 VIP CHRNA6'),
            "n48": ('n49', 'n50'),
            "n49": ('Inh L1-2 PAX6 CDH12', 'Inh L1-2 PAX6 TNFAIP8L3'),
            "n50": ('Inh L1 SST NMBR', 'n51'),
            "n51": ('n52', 'Inh L2-6 LAMP5 CA1'),
            "n52": ('Inh L1-4 LAMP5 LCP2', 'Inh L1-2 LAMP5 DBP'),
            "n53": ('n54', 'Inh L2-5 PVALB SCUBE3'),
            "n54": ('Inh L3-6 SST NPY', 'n55'),
            "n55": ('n61', 'n56'),
            "n56": ('Inh L5-6 GAD1 GLP1R', 'n57'),
            "n57": ('Inh L5-6 PVALB LGR5', 'n58'),
            "n58": ('n59', 'Inh L5-6 SST MIR548F2'),
            "n59": ('Inh L4-5 PVALB MEPE', 'n60'),
            "n60": ('Inh L2-4 PVALB WFDC2', 'Inh L4-6 PVALB SULF1'),
            "n61": ('n62', 'Inh L5-6 SST TH'),
            "n62": ('n65', 'n63'),
            "n63": ('n64', 'Inh L2-4 SST FRZB'),
            "n64": ('Inh L1-3 SST CALB1', 'Inh L3-5 SST ADGRG6'),
            "n65": ('Inh L3-6 SST HPGD', 'n66'),
            "n66": ('n67', 'Inh L4-5 SST STK32A'),
            "n67": ('n69', 'n68'),
            "n68": ('Inh L5-6 SST NPM1P10', 'Inh L4-6 SST GXYLT2'),
            "n69": ('Inh L4-6 SST B3GAT2', 'Inh L5-6 SST KLHDC8A'),
            "n70": ('n71', 'Micro L1-3 TYROBP'),
            "n71": ('n72', 'Endo L2-6 NOSTRIN'),
            "n72": ('n73', 'Oligo L1-6 OPALIN'),
            "n73": ('OPC L1-6 PDGFRA', 'n74'),
            "n74": ('Astro L1-6 FGFR3 SLC14A1', 'Astro L1-2 FGFR3 GFAP')
        }
        self.__checkClusterTree(tree_arr)
        self.__isFinished = True
        thread_build.join()
        time.sleep(0.2)

        self.__waitingMessage = "Reading data in..."
        self.__isFinished = False
        thread_read = threading.Thread(target=showProcess)
        thread_read.start()
        data = ad.read_h5ad(data_directory + "INPUT_DATA.h5ad")
        self.__dataSet = data
        self.__isFinished = True
        thread_read.join()
        time.sleep(0.2)
        return data

    # Read cells from a given directory.
    # data_directory: PathLike, representing the directory of the file, can be a ['pkl', 'csv', 'loom', 'h5ad'] file, e.g. "~/xhx/Python/neuron_full.pkl";
    #           if data_directory is None: use default data directory.
    # branch_name: str, representing the target branch, e.g. "n48";
    #           if branch_name is None: choose default branch; if default is still None, read the whole dataset.
    # cluster_set: a list or tuple of strings containing all target clusters to choose;
    # use_cluster_set: bool, indicating whether to activate choose from cluster_set;
    # file_type: can be one of ['pkl', 'csv', 'loom', 'h5ad'];
    # output: can be 'DataFrame' or 'AnnData', which indicates return type.
    # Return: a DataFrame or AnnData object.
    def readData(self, data_directory=None, branch_name=None, cluster_set=[], use_cluster_set=False, file_type=None, output=None):
        if data_directory is None:
            data_directory = self.__dataDirectory
        if not use_cluster_set and branch_name is None:
            branch_name = self.__branch
        data = None
        data_directory = data_directory.strip()
        if file_type is None:
            file_type = data_directory[-3:]
        if file_type == 'csv':
            try:
                data = pd.read_csv(data_directory)
            except:
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        elif file_type == 'pkl':
            try:
                data = pd.read_pickle(data_directory)
            except:
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        elif file_type == 'loom':
            try:
                data = ad.read_loom(data_directory)
            except:
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        elif file_type == 'h5ad':
            try:
                data = ad.read_h5ad(data_directory)
            except:
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        else:
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') receives an unrecognized file type.")
            return -1
        data = self.selectBranch(data, branch_name, cluster_set, use_cluster_set)
        if isinstance(data, int):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception.")
            return -1
        if output is None:
            return data
        elif output == "AnnData":
            if isinstance(data, pd.core.frame.DataFrame):
                return self.DataFrame_to_AnnData(data)
            else:
                return data
        elif output == "DataFrame":
            if isinstance(data, ad._core.anndata.AnnData):
                return self.AnnData_to_DataFrame(data)
            else:
                return data
        else:
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') receives a wrong output format parameter (must be 'AnnData' or 'DataFrame').")
            return -1
    
    # Select cells whose cluster is under the given branch or in given cluster set.
    # data: AnnData or DataFrame;
    # branch_name: str, representing the target branch, e.g. "n48";
    #           if branch_name is None: choose default branch; if default is still None, read the whole dataset.
    # cluster_set: a list or tuple of strings containing all target clusters to choose;
    # use_cluster_set: bool, indicating whether to activate choose from cluster_set.
    # Return: a DataFrame or AnnData object.
    def selectBranch(self, data=None, branch_name=None, cluster_set=[], use_cluster_set=False):
        isAnnData = False
        if data is None:
            data = self.__dataSet
        if branch_name is None:
            branch_name = self.__branch
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.selectBranch()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        if isinstance(data, ad._core.anndata.AnnData):
            isAnnData = True
            data = self.AnnData_to_DataFrame(data)
        if use_cluster_set:
            if (not isinstance(cluster_set, list) and not isinstance(cluster_set, tuple)) or len(cluster_set) == 0:
                cluster_set = self.__clusterSet
            data = data[data['cluster'].isin(cluster_set)]
        elif branch_name != None:
            clusters = self.listBranch(branch_name)
            if isinstance(clusters, int):
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.selectBranch()' (in file '" + __file__ + "') throws an exception.")
                return -1
            data = data[data['cluster'].isin(clusters)]
        if isAnnData:
            return self.DataFrame_to_AnnData(data)
        else:
            return data
    
    # Merge all clusters under a given branch.
    # data: AnnData or DataFrame;
    # branch_name: str, the clusters under the branch will merge, relabelled as the branch_name;
    # Return: AnnData or DataFrame.
    def mergeBranch(self, data=None, branch_name=None):
        isAnnData = False
        if data is None:
            data = self.__dataSet
        if branch_name is None:
            branch_name = self.__branch
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.mergeBranch()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        if isinstance(data, ad._core.anndata.AnnData):
            isAnnData = True
            data = self.AnnData_to_DataFrame(data)
        if branch_name != None:
            clusters = self.listBranch(branch_name)
            if isinstance(clusters, int):
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.mergeBranch()' (in file '" + __file__ + "') throws an exception.")
                return -1
            data.loc[data[data['cluster'].isin(clusters)].index.tolist(), 'cluster'] = branch_name
        if isAnnData:
            return self.DataFrame_to_AnnData(data)
        else:
            return data
        
    # Convert AnnData to DataFrame.
    # adata: an AnnData object.
    # Return: a DataFrame object.
    def AnnData_to_DataFrame(self, adata=None):
        if adata is None:
            adata = self.__dataSet
        if not isinstance(adata, ad._core.anndata.AnnData):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.AnnData_to_DataFrame()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData').")
            return -1
        return pd.concat([pd.DataFrame(adata.X, columns=adata.var.index.values).reset_index(drop=True), adata.obs.reset_index(drop=True)], axis=1, join="inner")

    # Convert DataFrame to AnnData.
    # data: a DataFrame object.
    # Return: an AnnData object.
    def DataFrame_to_AnnData(self, data=None):
        if data is None:
            data = self.__dataSet
        if not isinstance(data, pd.core.frame.DataFrame):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.DataFrame_to_AnnData()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'DataFrame').")
            return -1
        obs = pd.DataFrame(data["cluster"], columns=["cluster"])
        obs["cluster"] = obs.cluster.astype("category")
        data.drop(["cluster", "Unnamed: 0"], axis=1, inplace=True)
        var = pd.DataFrame(index=data.columns.values)
        X = np.array(data)
        return ad.AnnData(np.array(data), obs=obs, var=var, dtype="float")
    
    # Filter genes customly.
    # data: AnnData or DataFrame;
    # min_partial: float, to filter genes expressed in less than min_partial * 100% cells;
    #           if min_partial is None: do not filter.
    # gene_set: list or tuple, to filter genes appeared in gene_set;
    # gene_prefix: list or tuple, to filter genes with prefix in gene_prefix.
    # Return: a DataFrame or AnnData object.
    def geneFiltering(self, data=None, min_partial=None, gene_set=None, gene_prefix=None):
        isAnnData = False
        if data is None:
            data = self.__dataSet
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.geneFiltering()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        if isinstance(data, ad._core.anndata.AnnData):
            isAnnData = True
            data = self.AnnData_to_DataFrame(data)
        if isinstance(gene_set, list) or isinstance(gene_set, tuple):
            target = [item for item in data.columns.values if item in gene_set]
            data = data.drop(target, axis=1)
        if isinstance(gene_prefix, list) or isinstance(gene_prefix, tuple):
            def check(item):
                for x in gene_prefix:
                    if item.startswith(x):
                        return True
                return False
            target = [item for item in data.columns.values if check(item)]
            data = data.drop(target, axis=1)
        if isinstance(min_partial, float):
            target = []
            for idx, columns in data.iteritems():
                if idx != 'cluster':
                    expression = data[idx].to_numpy()
                    expression = expression[expression > 0]
                    if len(expression) / len(data) < min_partial:
                        target.append(idx)
            data = data.drop(target, axis=1)
        if isAnnData:
            return self.DataFrame_to_AnnData(data)
        else:
            return data

    # Do binary classification and generate shap figures.
    # data: an AnnData or DataFrame object;
    # cluster_name: str, the target cluster;
    # use_SMOTE: bool, indicates whether to use smote to oversample the data;
    # nthread: int, the number of running threads;
    # shap_params: dictionary, the shap plot parameters, indicating which kinds of figure to plot.
    def explainBinary(self, data=None, cluster_name=None, use_SMOTE=False, nthread=32, shap_params=None):
        def showProcess():
            print(self.__waitingMessage, end="  ")
            while not self.__isFinished:
                print('\b-', end='')
                time.sleep(0.05)
                print('\b\\', end='')
                time.sleep(0.05)
                print('\b|', end='')
                time.sleep(0.05)
                print('\b/', end='')
                time.sleep(0.05)
            print('\bdone')

        # Preprocessing data
        self.__waitingMessage = "Preprocessing data.."
        self.__isFinished = False
        thread_preprocessData = threading.Thread(target=showProcess)
        thread_preprocessData.start()
        if data is None:
            data = self.__dataSet
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            self.__isFinished = True
            thread_preprocessData.join()
            time.sleep(0.2)
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.explainBinary()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        if isinstance(data, ad._core.anndata.AnnData):
            data = self.AnnData_to_DataFrame(data)
        y = np.array(data['cluster'])
        x = data.drop(columns=['cluster'])
        if use_SMOTE:
            oversample = SMOTE()
            x, y = oversample.fit_resample(x, y)
        if cluster_name is None:
            cluster_name = self.__cluster
        y[y != cluster_name] = False
        y[y == cluster_name] = True
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)
        x_train = x_train.reset_index(drop=True)
        self.__isFinished = True
        thread_preprocessData.join()
        time.sleep(0.2)

        # Building the model
        self.__waitingMessage = "Building xgboost models.."
        self.__isFinished = False
        thread_buildModels = threading.Thread(target=showProcess)
        thread_buildModels.start()
        x_test = x_test.reset_index(drop=True)
        self.__XGBClassifer = XGBClassifier(objective="binary:logistic", nthread=nthread, eval_metric="mlogloss", random_state=42, use_label_encoder=False)
        self.__XGBClassifer.fit(x_train, y_train)
        self.__isFinished = True
        thread_buildModels.join()

        # Cross validation
        y_pred = self.__XGBClassifer.predict(x_test)
        accuracy = np.sum(y_pred == y_test) / len(y_pred) * 100
        print("Accuracy: %.4g%%" % accuracy)
        time.sleep(0.2)

        # Building the shap explainer
        self.__waitingMessage = "Building shap explainers.."
        self.__isFinished = False
        thread_buildShap = threading.Thread(target=showProcess)
        thread_buildShap.start()
        self.__explainer = shap.TreeExplainer(self.__XGBClassifer)
        self.__shapValues = self.__explainer.shap_values(x_test, approximate=True)
        self.__isFinished = True
        thread_buildShap.join()
        time.sleep(0.2)

        # Generating shap figures
        print("Generating shap figures..")
        if shap_params is None:
            shap_params = self.__shapParamsBinary
        if "max_display" not in shap_params or not isinstance(shap_params["max_display"], int):
            shap_params["max_display"] = 10
        self.__maxDisplay = shap_params["max_display"]
        self.__featureNames = x.columns.values
        if "bar_plot" in shap_params and shap_params["bar_plot"]:
            print("     Drawing bar plot..")
            plt.figure(1)
            plt.title("Target Cluster: " + cluster_name)
            shap.summary_plot(self.__shapValues, x_test, feature_names=self.__featureNames, max_display=self.__maxDisplay, plot_type='bar', show=False)
            plt.show()
        if "beeswarm" in shap_params and shap_params["beeswarm"]:
            print("     Drawing beeswarm plot..")
            plt.figure(2)
            plt.title("Target Cluster: " + cluster_name)
            shap.summary_plot(self.__shapValues, x_test, feature_names=self.__featureNames, max_display=self.__maxDisplay)
            plt.show()
        if "force_plot" in shap_params and shap_params["force_plot"]:
            print("     Drawing force plot..")
            print("     \033[1;33;40mWarning:\033[0m: force plot has not been stably supported yet.")
            shap.initjs()
            shap.plots.force(self.__explainer.expected_value, self.__shapValues, x_test, feature_names=self.__featureNames, show=False)
        if "heat_map" in shap_params and shap_params["heat_map"]:
            print("     Drawing heat map..")
            plt.figure(3)
            plt.title("Target Cluster: " + cluster_name)
            shap.plots.heatmap(self.__explainer(x_test), show=False)
            plt.show()
        if "decision_plot" in shap_params and shap_params["decision_plot"]:
            print("     Drawing decision plot..")
            plt.figure(4)
            plt.title("Target Cluster: " + cluster_name)
            y_pred = pd.DataFrame(y_pred).to_numpy()
            x_target = x_test[y_pred == 1]
            shap_values = self.__explainer.shap_values(x_target, approximate=True)
            shap.decision_plot(self.__explainer.expected_value, shap_values, x_target, link='logit', show=False)
            plt.show()

    # Do multi-classification and generate shap figures.
    # data: an AnnData or DataFrame object;
    # use_SMOTE: bool, indicates whether to use smote to oversample the data;
    # nthread: int, the number of running threads;
    # shap_params: dictionary, the shap plot parameters, indicating which kinds of figure to plot.
    def explainMulti(self, data=None, use_SMOTE=False, nthread=32, shap_params=None):
        def showProcess():
            print(self.__waitingMessage, end="  ")
            while not self.__isFinished:
                print('\b-', end='')
                time.sleep(0.05)
                print('\b\\', end='')
                time.sleep(0.05)
                print('\b|', end='')
                time.sleep(0.05)
                print('\b/', end='')
                time.sleep(0.05)
            print('\bdone')

        # Preprocessing data
        self.__waitingMessage = "Preprocessing data.."
        self.__isFinished = False
        thread_preprocessData = threading.Thread(target=showProcess)
        thread_preprocessData.start()
        if data is None:
            data = self.__dataSet
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            self.__isFinished = True
            thread_preprocessData.join()
            time.sleep(0.2)
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.explainMulti()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        if isinstance(data, ad._core.anndata.AnnData):
            data = self.AnnData_to_DataFrame(data)
        y = np.array(data['cluster'])
        x = data.drop(columns=['cluster'])
        if use_SMOTE:
            oversample = SMOTE()
            x, y = oversample.fit_resample(x, y)
        self.numOfClusters = 0
        self.clusterDict = {}
        [rows] = y.shape
        for i in range(rows):
            if y[i] in self.clusterDict:
                y[i] = self.clusterDict[y[i]]
            else:
                self.clusterDict[y[i]] = self.numOfClusters
                y[i] = self.numOfClusters
                self.numOfClusters += 1
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)
        x_train = x_train.reset_index(drop=True)
        self.__isFinished = True
        thread_preprocessData.join()
        time.sleep(0.2)
        for key in self.clusterDict.keys():
            print("     " + key + ": Class", self.clusterDict[key])

        # Building the model
        self.__waitingMessage = "Building xgboost models.."
        self.__isFinished = False
        thread_buildModels = threading.Thread(target=showProcess)
        thread_buildModels.start()
        x_test = x_test.reset_index(drop=True)
        self.__XGBClassifer = XGBClassifier(objective="multi:softmax", num_class=self.numOfClusters, nthread=nthread, eval_metric="mlogloss", random_state=42, use_label_encoder=False)
        self.__XGBClassifer.fit(x_train, y_train)
        self.__isFinished = True
        thread_buildModels.join()

        # Cross validation
        y_pred = self.__XGBClassifer.predict(x_test)
        accuracy = np.sum(y_pred == y_test) / len(y_pred) * 100
        print("Accuracy: %.4g%%" % accuracy)
        time.sleep(0.2)

        # Building the shap explainer
        self.__waitingMessage = "Building shap explainers.."
        self.__isFinished = False
        thread_buildShap = threading.Thread(target=showProcess)
        thread_buildShap.start()
        self.__explainer = shap.TreeExplainer(self.__XGBClassifer)
        self.__shapValues = self.__explainer.shap_values(x_test, approximate=True)
        self.__isFinished = True
        thread_buildShap.join()
        time.sleep(0.2)

        # Generating shap figures
        print("Generating shap figures..")
        if shap_params is None:
            shap_params = self.__shapParamsMulti
        if "max_display" not in shap_params or not isinstance(shap_params["max_display"], int):
            shap_params["max_display"] = 10
        self.__maxDisplay = shap_params["max_display"]
        self.__featureNames = x.columns.values
        if "bar_plot" in shap_params and shap_params["bar_plot"]:
            print("     Drawing bar plot..")
            plt.figure(1)
            shap.summary_plot(self.__shapValues, x_test, feature_names=self.__featureNames, max_display=self.__maxDisplay, show=False)
            plt.show()
        if "beeswarm" in shap_params and shap_params["beeswarm"]:
            print("     Drawing beeswarm plot..")
            print("     \033[1;33;40mWarning:\033[0m I am not sure whether there is a segementation fault (core dumped). If so, please contact the developer.")
            print("     \033[1;33;40mWarning:\033[0m There is a problem on text size of shap figures. See issue #995 at https://github.com/slundberg/shap/issues/995")
            figure = plt.figure(2)
            rows = self.numOfClusters // 2 + self.numOfClusters % 2
            cols = 2
            index = 1
            for key in self.clusterDict.keys():
                print("         Drawing cluster " + key + "...")
                figure_sub = figure.add_subplot(rows, cols, index)
                figure_sub.set_title("Target Cluster: " + key, fontsize=36)
                shap.summary_plot(self.__shapValues[self.clusterDict[key]], x_test, feature_names=self.__featureNames, max_display=self.__maxDisplay, show=False)
                index += 1
            figure.subplots_adjust(right=5, top=rows*3.5, hspace=0.2, wspace=0.2)
            plt.show()
        if "decision_plot" in shap_params and shap_params["decision_plot"]:
            print("     Drawing decision plot..")
            print("     \033[1;33;40mWarning:\033[0m I am not sure whether there is a segementation fault (core dumped). If so, please contact the developer.")
            print("     \033[1;33;40mWarning:\033[0m There is a problem on text size of shap figures. See issue #995 at https://github.com/slundberg/shap/issues/995")
            y_pred = pd.DataFrame(self.__XGBClassifer.predict_proba(x_test))
            figure = plt.figure(3)
            rows = self.numOfClusters // 2 + self.numOfClusters % 2
            cols = 2
            index = 1
            for key in self.clusterDict.keys():
                print("         Drawing cluster " + key + "...")
                y_pred_i = y_pred[y_pred.columns[self.clusterDict[key]]].to_numpy()
                x_target = x_test[y_pred_i >= 0.9]
                if len(x_target) == 0:
                    print("         \033[1;33;40mWarning:\033[0m empty dataset, skipped. Try setting 'use_SMOTE=True'.")
                    index -= 1
                    continue
                figure_sub = figure.add_subplot(rows, cols, index)
                figure_sub.set_title("Target Cluster: " + key, fontsize=36)
                shap.decision_plot(self.__explainer.expected_value[self.clusterDict[key]], self.__explainer.shap_values(x_target)[self.clusterDict[key]], x_target, link='logit', show=False)
                index += 1
            figure.subplots_adjust(right=5, top=rows*3.5, hspace=0.2, wspace=0.2)
            plt.show()
    
    def help(self, cmd=None):
        num_of_spaces = 110
        emptyline = ''
        if cmd == 'documentations' or cmd == 'apilist':
            documentations = '                                              \033[1;37;40mDocumentations\033[0m                                  '
            nameAndVersion = '                                            ' + __name__ + ': v' + __version__
            initialization = '\033[1;37;40mInitializations:\033[0m'
            sctreeshap = 'sctreeshap(): construct a sctreeshap object.'
            settings = '\033[1;37;40mSettings:\033[0m'
            setDataDirectory = 'setDataDirectory(): set default data directory.'
            setDataSet = 'setDataSet(): set default dataset.'
            setBranch = 'setBranch(): set default branch.'
            setCluster = 'setCluster(): set default cluster.'
            setClusterSet = 'setClusterSet(): set default target cluster set.'
            setShapParamsBinary = 'setShapParamsBinary(): set default shap plots parameters of explainBinary().'
            setShapParamsMulti = 'setShapParamsMulti(): set default shap plots parameters of explainMulti().'
            findCluster = 'findCluster(): find which branch a given cluster is in.'
            listBranch = 'listBranch(): list the clusters of a given branch.'
            loadDefault = 'loadDefault(): load default dataset and build default cluster tree.'
            dataprocessing = '\033[1;37;40mData processing:\033[0m'
            readData = 'readData(): read cells from a given directory.'
            selectBranch = 'selectBranch(): select cells whose cluster is under the given branch or in given cluster set.'
            mergeBranch = 'mergeBranch(): merge all clusters under a given branch.'
            AnnData_to_DataFrame = 'AnnData_to_DataFrame(): convert AnnData to DataFrame.'
            DataFrame_to_AnnData = 'DataFrame_to_AnnData(): convert DataFrame to AnnData.'
            geneFiltering = 'geneFiltering(): filter genes customly.'
            analysis = '\033[1;37;40mAnalysis:\033[0m'
            explainBinary = 'explainBinary(): do binary classification and generate shap figures.'
            explainMulti = 'explainMulti(): do multi-classification and generate shap figures.'
            getClassifier = "getClassifier(): get XGBClassifier of the last job (available after 'explainBinary()' or 'explainMulti()')."
            getExplainer = "getExplainer(): get shap explainer of the last job (available after 'explainBinary()' or 'explainMulti()')."
            getShapValues = "getShapValues(): get shap values of the last job (available after 'explainBinary()' or 'explainMulti()')."
            getTopGenes = "getTopGenes(): get top genes of max absolute mean shap values."
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + documentations + ' ' * (num_of_spaces - len(documentations) + 14) + '  |\n' \
                + '|  ' + nameAndVersion + ' ' * (num_of_spaces - len(nameAndVersion)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + initialization + ' ' * (num_of_spaces - len(initialization) + 14) + '  |\n' \
                + '|  ' + sctreeshap + ' ' * (num_of_spaces - len(sctreeshap)) + '  |\n' \
                + '|  ' + findCluster + ' ' * (num_of_spaces - len(findCluster)) + '  |\n' \
                + '|  ' + listBranch + ' ' * (num_of_spaces - len(listBranch)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + settings + ' ' * (num_of_spaces - len(settings) + 14) + '  |\n' \
                + '|  ' + setDataDirectory + ' ' * (num_of_spaces - len(setDataDirectory)) + '  |\n' \
                + '|  ' + setDataSet + ' ' * (num_of_spaces - len(setDataSet)) + '  |\n' \
                + '|  ' + setBranch + ' ' * (num_of_spaces - len(setBranch)) + '  |\n' \
                + '|  ' + setCluster + ' ' * (num_of_spaces - len(setCluster)) + '  |\n' \
                + '|  ' + setClusterSet + ' ' * (num_of_spaces - len(setClusterSet)) + '  |\n' \
                + '|  ' + setShapParamsBinary + ' ' * (num_of_spaces - len(setShapParamsBinary)) + '  |\n' \
                + '|  ' + setShapParamsMulti + ' ' * (num_of_spaces - len(setShapParamsMulti)) + '  |\n' \
                + '|  ' + loadDefault + ' ' * (num_of_spaces - len(loadDefault)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + dataprocessing + ' ' * (num_of_spaces - len(dataprocessing) + 14) + '  |\n' \
                + '|  ' + readData + ' ' * (num_of_spaces - len(readData)) + '  |\n' \
                + '|  ' + selectBranch + ' ' * (num_of_spaces - len(selectBranch)) + '  |\n' \
                + '|  ' + mergeBranch + ' ' * (num_of_spaces - len(mergeBranch)) + '  |\n' \
                + '|  ' + AnnData_to_DataFrame + ' ' * (num_of_spaces - len(AnnData_to_DataFrame)) + '  |\n' \
                + '|  ' + DataFrame_to_AnnData + ' ' * (num_of_spaces - len(DataFrame_to_AnnData)) + '  |\n' \
                + '|  ' + geneFiltering + ' ' * (num_of_spaces - len(geneFiltering)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + analysis + ' ' * (num_of_spaces - len(analysis) + 14) + '  |\n' \
                + '|  ' + explainBinary + ' ' * (num_of_spaces - len(explainBinary)) + '  |\n' \
                + '|  ' + explainMulti + ' ' * (num_of_spaces - len(explainMulti)) + '  |\n' \
                + '|  ' + getShapValues + ' ' * (num_of_spaces - len(getShapValues)) + '  |\n' \
                + '|  ' + getTopGenes + ' ' * (num_of_spaces - len(getTopGenes)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'sctreeshap':
            function = '\033[1;37;40msctreeshap.sctreeshap\033[0m'
            api = 'class sctreeshap.sctreeshap(tree_arr=None)'
            description =               'Description:   Construct a sctreeshap object.'
            tree_arr =                  'Parameters:    tree_arr: dictionary'
            tree_arr_description1 =     '               |  tree_arr[str] represents the node of name str in the cluster tree, can be a list or a tuple'
            tree_arr_description2 =     '               |  of strings, representing the name of childs of the node (from left to right).'
            tree_arr_description3 =     "               |  e.g. tree_arr['n1'] = ('n2', 'n70') represents a node named 'n1', whose left child is 'n2' "
            tree_arr_description4 =     "               |  and right child is 'n70'."
            tree_arr_description5 =     '               |  Note that you do not need to create nodes for clusters, since they are leaf nodes and have'
            tree_arr_description6 =     '               |  no childs.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + tree_arr + ' ' * (num_of_spaces - len(tree_arr)) + '  |\n' \
                + '|  ' + tree_arr_description1 + ' ' * (num_of_spaces - len(tree_arr_description1)) + '  |\n' \
                + '|  ' + tree_arr_description2 + ' ' * (num_of_spaces - len(tree_arr_description2)) + '  |\n' \
                + '|  ' + tree_arr_description3 + ' ' * (num_of_spaces - len(tree_arr_description3)) + '  |\n' \
                + '|  ' + tree_arr_description4 + ' ' * (num_of_spaces - len(tree_arr_description4)) + '  |\n' \
                + '|  ' + tree_arr_description5 + ' ' * (num_of_spaces - len(tree_arr_description5)) + '  |\n' \
                + '|  ' + tree_arr_description6 + ' ' * (num_of_spaces - len(tree_arr_description6)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setDataDirectory':
            function = '\033[1;37;40msctreeshap.sctreeshap.setDataDirectory\033[0m'
            api = 'sctreeshap.sctreeshap.setDataDirectory(data_directory=None)'
            description =                   'Description: set default data directory.'
            data_directory =                'Parameters:  data_directory: PathLike'
            data_directory_description1 =   '             |  The directory of default input file, can be a .pkl file or a .csv file.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data_directory + ' ' * (num_of_spaces - len(data_directory)) + '  |\n' \
                + '|  ' + data_directory_description1 + ' ' * (num_of_spaces - len(data_directory_description1)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setDataSet':
            function = '\033[1;37;40msctreeshap.sctreeshap.setDataSet\033[0m'
            api = 'sctreeshap.sctreeshap.setDataSet(data=None)'
            description =                   'Description: set default dataset.'
            data =                          'Parameters:  data: DataFrame or AnnData'
            data_description1 =             '             |  The default dataset.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                + '|  ' + data_description1 + ' ' * (num_of_spaces - len(data_description1)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setBranch':
            function = '\033[1;37;40msctreeshap.sctreeshap.setBranch\033[0m'
            api = 'sctreeshap.sctreeshap.setBranch(branch_name=None)'
            description =                   'Description: set default branch.'
            branch_name =                          'Parameters:  branch_name: str'
            branch_name_description1 =             '             |  The default branch.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setCluster':
            function = '\033[1;37;40msctreeshap.sctreeshap.setCluster\033[0m'
            api = 'sctreeshap.sctreeshap.setCluster(cluster_name=None)'
            description =                   'Description: set default cluster.'
            cluster_name =                          'Parameters:  cluster_name: str'
            cluster_name_description1 =             '             |  The default cluster for binary classification.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + cluster_name + ' ' * (num_of_spaces - len(cluster_name)) + '  |\n' \
                + '|  ' + cluster_name_description1 + ' ' * (num_of_spaces - len(cluster_name_description1)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setClusterSet':
            function = '\033[1;37;40msctreeshap.sctreeshap.setClusterSet\033[0m'
            api = 'sctreeshap.sctreeshap.setClusterSet(cluster_set=None)'
            description =                   'Description: set default target cluster set.'
            cluster_set =                          'Parameters:  cluster_set: list or tuple'
            cluster_set_description1 =             '             |  A list or tuple of strings to select data whose cluster is within it.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + cluster_set + ' ' * (num_of_spaces - len(cluster_set)) + '  |\n' \
                + '|  ' + cluster_set_description1 + ' ' * (num_of_spaces - len(cluster_set_description1)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setShapParamsBinary':
            function = '\033[1;37;40msctreeshap.sctreeshap.setShapParamsBinary\033[0m'
            api = 'sctreeshap.sctreeshap.setShapParamsBinary(shap_params=None)'
            description =                   'Description: set default shap plots parameters of explainBinary().'
            shap_params =                          'Parameters:  shap_params: dictionary'
            shap_params_description1 =             '             |  Keys: ["max_display", "bar_plot", "beeswarm", "force_plot", "heat_map", "decision_plot"]'
            shap_params_description2 =             '             |  Default values: [10, True, True, False, False, False]'
            shap_params_description3 =             '             |  Determine what kinds of figures to output, and maximum number of genes you want to depict.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'setShapParamsMulti':
            function = '\033[1;37;40msctreeshap.sctreeshap.setShapParamsMulti\033[0m'
            api = 'sctreeshap.sctreeshap.setShapParamsMulti(shap_params=None)'
            description =                   'Description: set default shap plots parameters of explainMulti().'
            shap_params =                          'Parameters:  shap_params: dictionary'
            shap_params_description1 =             '             |  Keys: ["max_display", "bar_plot", "beeswarm", "decision_plot"]'
            shap_params_description2 =             '             |  Default values: [10, True, False, False]'
            shap_params_description3 =             '             |  Determine what kinds of figures to output, and maximum number of genes you want to depict.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'findCluster':
            function = '\033[1;37;40msctreeshap.sctreeshap.findCluster\033[0m'
            api = 'sctreeshap.sctreeshap.findCluster(cluster_name=None)'
            description =                   'Description: find which branch a given cluster is in.'
            cluster_name =                          'Parameters:  cluster_name: str'
            cluster_name_description1 =             '             |  The target cluster.'
            return_description =                    'Return:      str, the path from root to the cluster.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + cluster_name + ' ' * (num_of_spaces - len(cluster_name)) + '  |\n' \
                + '|  ' + cluster_name_description1 + ' ' * (num_of_spaces - len(cluster_name_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'listBranch':
            function = '\033[1;37;40msctreeshap.sctreeshap.listBranch\033[0m'
            api = 'sctreeshap.sctreeshap.listBranch(branch_name=None)'
            description =                   'Description: list the clusters of a given branch'
            branch_name =                          'Parameters:  branch_name: str'
            branch_name_description1 =             '             |  The target branch.'
            return_description =                    'Return:      list, all clusters under the branch.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'readData':
            function = '\033[1;37;40msctreeshap.sctreeshap.readData\033[0m'
            api = "sctreeshap.sctreeshap.readData(data_directory=None, branch_name=None, cluster_set=[], use_cluster_set=False, "
            api1 = "file_type=None, output=None)"
            description =                   'Description: read cells from a given directory.'
            data_directory =                          'Parameters:  data_directory: PathLike'
            data_directory_description1 =             "             |  The directory of the input file, can be a ['pkl', 'csv', 'loom', 'h5ad'] file."
            branch_name =                             '             branch_name: str'
            branch_name_description1 =                '             |  If not None, filter cells not under the branch.'
            cluster_set =                             '             cluster_set: list or tuple'
            cluster_set_description1 =                '             |  A list or tuple of strings representing the target clusters.'
            use_cluster_set =                         '             use_cluster_set: bool'
            use_cluster_set_description1 =            '             |  If True, filter cells not with clusters in cluster_set.'
            file_type =                               '             file_type: str'
            file_type_description1 =                  "             |  Can be one of ['pkl', 'csv', 'loom', 'h5ad']."
            output =                                  "             output: 'DataFrame' or 'AnnData'"
            output_description1 =                     '             |  Determine the return type of the function.'
            return_description =                      'Return:      AnnData or DataFrame.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + api1 + ' ' * (num_of_spaces - len(api1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data_directory + ' ' * (num_of_spaces - len(data_directory)) + '  |\n' \
                + '|  ' + data_directory_description1 + ' ' * (num_of_spaces - len(data_directory_description1)) + '  |\n' \
                + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                + '|  ' + cluster_set + ' ' * (num_of_spaces - len(cluster_set)) + '  |\n' \
                + '|  ' + cluster_set_description1 + ' ' * (num_of_spaces - len(cluster_set_description1)) + '  |\n' \
                + '|  ' + use_cluster_set + ' ' * (num_of_spaces - len(use_cluster_set)) + '  |\n' \
                + '|  ' + use_cluster_set_description1 + ' ' * (num_of_spaces - len(use_cluster_set_description1)) + '  |\n' \
                + '|  ' + file_type + ' ' * (num_of_spaces - len(file_type)) + '  |\n' \
                + '|  ' + file_type_description1 + ' ' * (num_of_spaces - len(file_type_description1)) + '  |\n' \
                + '|  ' + output + ' ' * (num_of_spaces - len(output)) + '  |\n' \
                + '|  ' + output_description1 + ' ' * (num_of_spaces - len(output_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'mergeBranch':
            function = '\033[1;37;40msctreeshap.sctreeshap.mergeBranch\033[0m'
            api = 'sctreeshap.sctreeshap.mergeBranch(data=None, branch_name=None)'
            description =                    'Description: merge all clusters under a given branch.'
            data =                          'Parameters:  data: AnnData or DataFrame'
            branch_name =                   '             branch_name: str'
            branch_name_description1 =      '             |  The clusters under the branch will merge, relabelled as the branch_name.'
            return_description =            'Return:      DataFrame or AnnData.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'AnnData_to_DataFrame':
            function = '\033[1;37;40msctreeshap.sctreeshap.AnnData_to_DataFrame\033[0m'
            api = 'sctreeshap.sctreeshap.AnnData_to_DataFrame(adata)'
            description =                   'Description: convert AnnData to DataFrame.'
            adata =                          'Parameters:  adata: AnnData'
            adata_description1 =             '             |  An AnnData object in anndata package.'
            return_description =             'Return:      DataFrame.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + adata + ' ' * (num_of_spaces - len(adata)) + '  |\n' \
                + '|  ' + adata_description1 + ' ' * (num_of_spaces - len(adata_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'DataFrame_to_AnnData':
            function = '\033[1;37;40msctreeshap.sctreeshap.DataFrame_to_AnnData\033[0m'
            api = 'sctreeshap.sctreeshap.DataFrame_to_AnnData(data)'
            description =                    'Description: convert DataFrame to AnnData.'
            data =                          'Parameters:  data: DataFrame'
            data_description1 =             '             |  A DataFrame object in pandas package.'
            return_description =             'Return:      AnnData.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                + '|  ' + data_description1 + ' ' * (num_of_spaces - len(data_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'geneFiltering':
            function = '\033[1;37;40msctreeshap.sctreeshap.geneFiltering\033[0m'
            api = 'sctreeshap.sctreeshap.geneFiltering(data=None, min_partial=None, gene_set=None, gene_prefix=None)'
            description =                    'Description: filter genes customly.'
            data =                          'Parameters:  data: DataFrame or AnnData'
            min_partial =                   '             min_partial: float'
            min_partial_description1 =      '             |  If not None, filter genes expressed in less than min_partial * 100% cells.'
            gene_set =                      '             gene_set: list or tuple'
            gene_set_description1 =         '             |  A list or a tuple of genes to be filtered.'
            gene_prefix =                   '             gene_prefix: list or tuple'
            gene_prefix_description1 =      '             |  Genes with prefix in gene_prefix will be filtered.'
            return_description =             'Return:      AnnData or DataFrame.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                + '|  ' + min_partial + ' ' * (num_of_spaces - len(min_partial)) + '  |\n' \
                + '|  ' + min_partial_description1 + ' ' * (num_of_spaces - len(min_partial_description1)) + '  |\n' \
                + '|  ' + gene_set + ' ' * (num_of_spaces - len(gene_set)) + '  |\n' \
                + '|  ' + gene_set_description1 + ' ' * (num_of_spaces - len(gene_set_description1)) + '  |\n' \
                + '|  ' + gene_prefix + ' ' * (num_of_spaces - len(gene_prefix)) + '  |\n' \
                + '|  ' + gene_prefix_description1 + ' ' * (num_of_spaces - len(gene_prefix_description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'explainBinary':
            function = '\033[1;37;40msctreeshap.sctreeshap.explainBinary\033[0m'
            api = 'sctreeshap.sctreeshap.explainBinary(data=None, cluster_name=None, use_SMOTE=False, nthread=32, '
            api1 = 'shap_params=None)'
            description =                    'Description: do binary classification and generate shap figures.'
            data =                          'Parameters:  data: DataFrame or AnnData'
            cluster_name =                  '             cluster_name: str'
            cluster_name_description1 =     '             |  The target cluster for classification.'
            use_SMOTE =                     '             use_SMOTE: bool'
            use_SMOTE_description1 =        '             |  True if you want to use SMOTE to resample.'
            nthread =                       '             nthread: int'
            nthread_description1 =          '             |  The number of running threads.'
            shap_params =                   '             shap_params: dictionary'
            shap_params_description1 =      '             |  Keys: ["max_display", "bar_plot", "beeswarm", "force_plot", "heat_map", "decision_plot"]'
            shap_params_description2 =      '             |  Values: "max_display" reflects an int, which determines the maximum number of genes to display'
            shap_params_description3 =      '             |  in shap figures, while each other key is reflected to a bool which indicates whether to '
            shap_params_description4 =      '             |  output this kind of shap figures.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + api1 + ' ' * (num_of_spaces - len(api1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                + '|  ' + cluster_name + ' ' * (num_of_spaces - len(cluster_name)) + '  |\n' \
                + '|  ' + cluster_name_description1 + ' ' * (num_of_spaces - len(cluster_name_description1)) + '  |\n' \
                + '|  ' + use_SMOTE + ' ' * (num_of_spaces - len(use_SMOTE)) + '  |\n' \
                + '|  ' + use_SMOTE_description1 + ' ' * (num_of_spaces - len(use_SMOTE_description1)) + '  |\n' \
                + '|  ' + nthread + ' ' * (num_of_spaces - len(nthread)) + '  |\n' \
                + '|  ' + nthread_description1 + ' ' * (num_of_spaces - len(nthread_description1)) + '  |\n' \
                + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                + '|  ' + shap_params_description4 + ' ' * (num_of_spaces - len(shap_params_description4)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'explainMulti':
            function = '\033[1;37;40msctreeshap.sctreeshap.explainMulti\033[0m'
            api = 'sctreeshap.sctreeshap.explainMulti(data=None, use_SMOTE=False, nthread=32, shap_params=None)'
            description =                    'Description: do multi-classification and generate shap figures.'
            data =                          'Parameters:  data: DataFrame or AnnData'
            use_SMOTE =                     '             use_SMOTE: bool'
            use_SMOTE_description1 =        '             |  True if you want to use SMOTE to resample.'
            nthread =                       '             nthread: int'
            nthread_description1 =          '             |  The number of running threads.'
            shap_params =                   '             shap_params: dictionary'
            shap_params_description1 =      '             |  Keys: ["max_display", "bar_plot", "beeswarm", "decision_plot"]'
            shap_params_description2 =      '             |  Values: "max_display" reflects an int, which determines the maximum number of genes to display'
            shap_params_description3 =      '             |  in shap figures, while each other key is reflected to a bool which indicates whether to '
            shap_params_description4 =      '             |  output this kind of shap figures.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                + '|  ' + use_SMOTE + ' ' * (num_of_spaces - len(use_SMOTE)) + '  |\n' \
                + '|  ' + use_SMOTE_description1 + ' ' * (num_of_spaces - len(use_SMOTE_description1)) + '  |\n' \
                + '|  ' + nthread + ' ' * (num_of_spaces - len(nthread)) + '  |\n' \
                + '|  ' + nthread_description1 + ' ' * (num_of_spaces - len(nthread_description1)) + '  |\n' \
                + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                + '|  ' + shap_params_description4 + ' ' * (num_of_spaces - len(shap_params_description4)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'getShapValues':
            function = '\033[1;37;40msctreeshap.sctreeshap.getShapValues\033[0m'
            api = 'sctreeshap.sctreeshap.getShapValues()'
            description =                    "Description: get shap values of the last job (available after 'a.explainBinary()' or 'a.explainMulti()')."
            return_description =             'Return:      ndarray.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd == 'getTopGenes':
            function = '\033[1;37;40msctreeshap.sctreeshap.getTopGenes\033[0m'
            api = 'sctreeshap.sctreeshap.getTopGenes(max_display=None, shap_values=None, feature_names=None)'
            description =                    'Description: get top genes of max absolute mean shap values. (You can just simply call it without any '
            description1 =                   '             parameters after explainBinary() or explainMulti()).'
            max_display =                    'Parameters:  max_display: int'
            max_display_description1 =       '             |  The the number of top genes you want to derive.'
            shap_values =                    '             shap_values: list or ndarray'
            shap_values_description1 =       '             |  This can be derived from getShapValues(), defaultly from last explainBinary()/explainMulti().'
            feature_names =                  '             feature_names: ndarray, list or tuple'
            feature_names_description1 =     '             |  The gene set (columns of cell-gene matrix), must correspond to order of shap_values, defaultly'
            feature_names_description2 =     '             |  from last explainBinary()/explainMulti().'
            return_description =             'Return:      ndarray, an ordered set of top genes of maximum absolute mean shap values.'
            return ' __' + '_' * num_of_spaces + '__ \n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                + '|  ' + description1 + ' ' * (num_of_spaces - len(description1)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + max_display + ' ' * (num_of_spaces - len(max_display)) + '  |\n' \
                + '|  ' + max_display_description1 + ' ' * (num_of_spaces - len(max_display_description1)) + '  |\n' \
                + '|  ' + shap_values + ' ' * (num_of_spaces - len(shap_values)) + '  |\n' \
                + '|  ' + shap_values_description1 + ' ' * (num_of_spaces - len(shap_values_description1)) + '  |\n' \
                + '|  ' + feature_names + ' ' * (num_of_spaces - len(feature_names)) + '  |\n' \
                + '|  ' + feature_names_description1 + ' ' * (num_of_spaces - len(feature_names_description1)) + '  |\n' \
                + '|  ' + feature_names_description2 + ' ' * (num_of_spaces - len(feature_names_description2)) + '  |\n' \
                + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                + '|__' + '_' * num_of_spaces + '__|'
        if cmd != None:
            return "Function " + cmd + " not found!"
        while True:
            print('* ', end='')
            cmd = input().strip()
            if cmd[:4] == 'EXIT':
                return None
            elif cmd[:4] == 'SHOW':
                cmd = cmd[4:].strip()
                if cmd == 'documentations' or cmd == 'apilist':
                    documentations = '                                              \033[1;37;40mDocumentations\033[0m                                  '
                    nameAndVersion = '                                            ' + __name__ + ': v' + __version__
                    initialization = '\033[1;37;40mInitializations:\033[0m'
                    sctreeshap = 'sctreeshap(): construct a sctreeshap object.'
                    settings = '\033[1;37;40mSettings:\033[0m'
                    setDataDirectory = 'setDataDirectory(): set default data directory.'
                    setDataSet = 'setDataSet(): set default dataset.'
                    setBranch = 'setBranch(): set default branch.'
                    setCluster = 'setCluster(): set default cluster.'
                    setClusterSet = 'setClusterSet(): set default target cluster set.'
                    setShapParamsBinary = 'setShapParamsBinary(): set default shap plots parameters of explainBinary().'
                    setShapParamsMulti = 'setShapParamsMulti(): set default shap plots parameters of explainMulti().'
                    findCluster = 'findCluster(): find which branch a given cluster is in.'
                    listBranch = 'listBranch(): list the clusters of a given branch.'
                    loadDefault = 'loadDefault(): load default dataset and build default cluster tree.'
                    dataprocessing = '\033[1;37;40mData processing:\033[0m'
                    readData = 'readData(): read cells from a given directory.'
                    selectBranch = 'selectBranch(): select cells whose cluster is under the given branch or in given cluster set.'
                    mergeBranch = 'mergeBranch(): merge all clusters under a given branch.'
                    AnnData_to_DataFrame = 'AnnData_to_DataFrame(): convert AnnData to DataFrame.'
                    DataFrame_to_AnnData = 'DataFrame_to_AnnData(): convert DataFrame to AnnData.'
                    geneFiltering = 'geneFiltering(): filter genes customly.'
                    analysis = '\033[1;37;40mAnalysis:\033[0m'
                    explainBinary = 'explainBinary(): do binary classification and generate shap figures.'
                    explainMulti = 'explainMulti(): do multi-classification and generate shap figures.'
                    getClassifier = "getClassifier(): get XGBClassifier of the last job (available after 'explainBinary()' or 'explainMulti()')."
                    getExplainer = "getExplainer(): get shap explainer of the last job (available after 'explainBinary()' or 'explainMulti()')."
                    getShapValues = "getShapValues(): get shap values of the last job (available after 'explainBinary()' or 'explainMulti()')."
                    getTopGenes = "getTopGenes(): get top genes of max absolute mean shap values."
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + documentations + ' ' * (num_of_spaces - len(documentations) + 14) + '  |\n' \
                        + '|  ' + nameAndVersion + ' ' * (num_of_spaces - len(nameAndVersion)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + initialization + ' ' * (num_of_spaces - len(initialization) + 14) + '  |\n' \
                        + '|  ' + sctreeshap + ' ' * (num_of_spaces - len(sctreeshap)) + '  |\n' \
                        + '|  ' + findCluster + ' ' * (num_of_spaces - len(findCluster)) + '  |\n' \
                        + '|  ' + listBranch + ' ' * (num_of_spaces - len(listBranch)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + settings + ' ' * (num_of_spaces - len(settings) + 14) + '  |\n' \
                        + '|  ' + setDataDirectory + ' ' * (num_of_spaces - len(setDataDirectory)) + '  |\n' \
                        + '|  ' + setDataSet + ' ' * (num_of_spaces - len(setDataSet)) + '  |\n' \
                        + '|  ' + setBranch + ' ' * (num_of_spaces - len(setBranch)) + '  |\n' \
                        + '|  ' + setCluster + ' ' * (num_of_spaces - len(setCluster)) + '  |\n' \
                        + '|  ' + setClusterSet + ' ' * (num_of_spaces - len(setClusterSet)) + '  |\n' \
                        + '|  ' + setShapParamsBinary + ' ' * (num_of_spaces - len(setShapParamsBinary)) + '  |\n' \
                        + '|  ' + setShapParamsMulti + ' ' * (num_of_spaces - len(setShapParamsMulti)) + '  |\n' \
                        + '|  ' + loadDefault + ' ' * (num_of_spaces - len(loadDefault)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + dataprocessing + ' ' * (num_of_spaces - len(dataprocessing) + 14) + '  |\n' \
                        + '|  ' + readData + ' ' * (num_of_spaces - len(readData)) + '  |\n' \
                        + '|  ' + selectBranch + ' ' * (num_of_spaces - len(selectBranch)) + '  |\n' \
                        + '|  ' + mergeBranch + ' ' * (num_of_spaces - len(mergeBranch)) + '  |\n' \
                        + '|  ' + AnnData_to_DataFrame + ' ' * (num_of_spaces - len(AnnData_to_DataFrame)) + '  |\n' \
                        + '|  ' + DataFrame_to_AnnData + ' ' * (num_of_spaces - len(DataFrame_to_AnnData)) + '  |\n' \
                        + '|  ' + geneFiltering + ' ' * (num_of_spaces - len(geneFiltering)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + analysis + ' ' * (num_of_spaces - len(analysis) + 14) + '  |\n' \
                        + '|  ' + explainBinary + ' ' * (num_of_spaces - len(explainBinary)) + '  |\n' \
                        + '|  ' + explainMulti + ' ' * (num_of_spaces - len(explainMulti)) + '  |\n' \
                        # + '|  ' + getClassifier + ' ' * (num_of_spaces - len(getClassifier)) + '  |\n' \
                        # + '|  ' + getExplainer + ' ' * (num_of_spaces - len(getExplainer)) + '  |\n' \
                        + '|  ' + getShapValues + ' ' * (num_of_spaces - len(getShapValues)) + '  |\n' \
                        + '|  ' + getTopGenes + ' ' * (num_of_spaces - len(getTopGenes)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'sctreeshap':
                    function = '\033[1;37;40msctreeshap.sctreeshap\033[0m'
                    api = 'class sctreeshap.sctreeshap(tree_arr=None)'
                    description =               'Description:   Construct a sctreeshap object.'
                    tree_arr =                  'Parameters:    tree_arr: dictionary'
                    tree_arr_description1 =     '               |  tree_arr[str] represents the node of name str in the cluster tree, can be a list or a tuple'
                    tree_arr_description2 =     '               |  of strings, representing the name of childs of the node (from left to right).'
                    tree_arr_description3 =     "               |  e.g. tree_arr['n1'] = ('n2', 'n70') represents a node named 'n1', whose left child is 'n2' "
                    tree_arr_description4 =     "               |  and right child is 'n70'."
                    tree_arr_description5 =     '               |  Note that you do not need to create nodes for clusters, since they are leaf nodes and have'
                    tree_arr_description6 =     '               |  no childs.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + tree_arr + ' ' * (num_of_spaces - len(tree_arr)) + '  |\n' \
                        + '|  ' + tree_arr_description1 + ' ' * (num_of_spaces - len(tree_arr_description1)) + '  |\n' \
                        + '|  ' + tree_arr_description2 + ' ' * (num_of_spaces - len(tree_arr_description2)) + '  |\n' \
                        + '|  ' + tree_arr_description3 + ' ' * (num_of_spaces - len(tree_arr_description3)) + '  |\n' \
                        + '|  ' + tree_arr_description4 + ' ' * (num_of_spaces - len(tree_arr_description4)) + '  |\n' \
                        + '|  ' + tree_arr_description5 + ' ' * (num_of_spaces - len(tree_arr_description5)) + '  |\n' \
                        + '|  ' + tree_arr_description6 + ' ' * (num_of_spaces - len(tree_arr_description6)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setDataDirectory':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setDataDirectory\033[0m'
                    api = 'sctreeshap.sctreeshap.setDataDirectory(data_directory=None)'
                    description =                   'Description: set default data directory.'
                    data_directory =                'Parameters:  data_directory: PathLike'
                    data_directory_description1 =   '             |  The directory of default input file, can be a .pkl file or a .csv file.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data_directory + ' ' * (num_of_spaces - len(data_directory)) + '  |\n' \
                        + '|  ' + data_directory_description1 + ' ' * (num_of_spaces - len(data_directory_description1)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setDataSet':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setDataSet\033[0m'
                    api = 'sctreeshap.sctreeshap.setDataSet(data=None)'
                    description =                   'Description: set default dataset.'
                    data =                          'Parameters:  data: DataFrame or AnnData'
                    data_description1 =             '             |  The default dataset.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                        + '|  ' + data_description1 + ' ' * (num_of_spaces - len(data_description1)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setBranch':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setBranch\033[0m'
                    api = 'sctreeshap.sctreeshap.setBranch(branch_name=None)'
                    description =                   'Description: set default branch.'
                    branch_name =                          'Parameters:  branch_name: str'
                    branch_name_description1 =             '             |  The default branch.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                        + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setCluster':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setCluster\033[0m'
                    api = 'sctreeshap.sctreeshap.setCluster(cluster_name=None)'
                    description =                   'Description: set default cluster.'
                    cluster_name =                          'Parameters:  cluster_name: str'
                    cluster_name_description1 =             '             |  The default cluster for binary classification.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + cluster_name + ' ' * (num_of_spaces - len(cluster_name)) + '  |\n' \
                        + '|  ' + cluster_name_description1 + ' ' * (num_of_spaces - len(cluster_name_description1)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setClusterSet':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setClusterSet\033[0m'
                    api = 'sctreeshap.sctreeshap.setClusterSet(cluster_set=None)'
                    description =                   'Description: set default target cluster set.'
                    cluster_set =                          'Parameters:  cluster_set: list or tuple'
                    cluster_set_description1 =             '             |  A list or tuple of strings to select data whose cluster is within it.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + cluster_set + ' ' * (num_of_spaces - len(cluster_set)) + '  |\n' \
                        + '|  ' + cluster_set_description1 + ' ' * (num_of_spaces - len(cluster_set_description1)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setShapParamsBinary':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setShapParamsBinary\033[0m'
                    api = 'sctreeshap.sctreeshap.setShapParamsBinary(shap_params=None)'
                    description =                   'Description: set default shap plots parameters of explainBinary().'
                    shap_params =                          'Parameters:  shap_params: dictionary'
                    shap_params_description1 =             '             |  Keys: ["max_display", "bar_plot", "beeswarm", "force_plot", "heat_map", "decision_plot"]'
                    shap_params_description2 =             '             |  Default values: [10, True, True, False, False, False]'
                    shap_params_description3 =             '             |  Determine what kinds of figures to output, and maximum number of genes you want to depict.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                        + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                        + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                        + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'setShapParamsMulti':
                    function = '\033[1;37;40msctreeshap.sctreeshap.setShapParamsMulti\033[0m'
                    api = 'sctreeshap.sctreeshap.setShapParamsMulti(shap_params=None)'
                    description =                   'Description: set default shap plots parameters of explainMulti().'
                    shap_params =                          'Parameters:  shap_params: dictionary'
                    shap_params_description1 =             '             |  Keys: ["max_display", "bar_plot", "beeswarm", "decision_plot"]'
                    shap_params_description2 =             '             |  Default values: [10, True, False, False]'
                    shap_params_description3 =             '             |  Determine what kinds of figures to output, and maximum number of genes you want to depict.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                        + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                        + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                        + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'findCluster':
                    function = '\033[1;37;40msctreeshap.sctreeshap.findCluster\033[0m'
                    api = 'sctreeshap.sctreeshap.findCluster(cluster_name=None)'
                    description =                   'Description: find which branch a given cluster is in.'
                    cluster_name =                          'Parameters:  cluster_name: str'
                    cluster_name_description1 =             '             |  The target cluster.'
                    return_description =                    'Return:      str, the path from root to the cluster.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + cluster_name + ' ' * (num_of_spaces - len(cluster_name)) + '  |\n' \
                        + '|  ' + cluster_name_description1 + ' ' * (num_of_spaces - len(cluster_name_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'listBranch':
                    function = '\033[1;37;40msctreeshap.sctreeshap.listBranch\033[0m'
                    api = 'sctreeshap.sctreeshap.listBranch(branch_name=None)'
                    description =                   'Description: list the clusters of a given branch'
                    branch_name =                          'Parameters:  branch_name: str'
                    branch_name_description1 =             '             |  The target branch.'
                    return_description =                    'Return:      list, all clusters under the branch.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                        + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'readData':
                    function = '\033[1;37;40msctreeshap.sctreeshap.readData\033[0m'
                    api = "sctreeshap.sctreeshap.readData(data_directory=None, branch_name=None, cluster_set=[], use_cluster_set=False, "
                    api1 = "file_type=None, output=None)"
                    description =                   'Description: read cells from a given directory.'
                    data_directory =                          'Parameters:  data_directory: PathLike'
                    data_directory_description1 =             "             |  The directory of the input file, can be a ['pkl', 'csv', 'loom', 'h5ad'] file."
                    branch_name =                             '             branch_name: str'
                    branch_name_description1 =                '             |  If not None, filter cells not under the branch.'
                    cluster_set =                             '             cluster_set: list or tuple'
                    cluster_set_description1 =                '             |  A list or tuple of strings representing the target clusters.'
                    use_cluster_set =                         '             use_cluster_set: bool'
                    use_cluster_set_description1 =            '             |  If True, filter cells not with clusters in cluster_set.'
                    file_type =                               '             file_type: str'
                    file_type_description1 =                  "             |  Can be one of ['pkl', 'csv', 'loom', 'h5ad']."
                    output =                                  "             output: 'DataFrame' or 'AnnData'"
                    output_description1 =                     '             |  Determine the return type of the function.'
                    return_description =                      'Return:      AnnData or DataFrame.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + api1 + ' ' * (num_of_spaces - len(api1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data_directory + ' ' * (num_of_spaces - len(data_directory)) + '  |\n' \
                        + '|  ' + data_directory_description1 + ' ' * (num_of_spaces - len(data_directory_description1)) + '  |\n' \
                        + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                        + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                        + '|  ' + cluster_set + ' ' * (num_of_spaces - len(cluster_set)) + '  |\n' \
                        + '|  ' + cluster_set_description1 + ' ' * (num_of_spaces - len(cluster_set_description1)) + '  |\n' \
                        + '|  ' + use_cluster_set + ' ' * (num_of_spaces - len(use_cluster_set)) + '  |\n' \
                        + '|  ' + use_cluster_set_description1 + ' ' * (num_of_spaces - len(use_cluster_set_description1)) + '  |\n' \
                        + '|  ' + file_type + ' ' * (num_of_spaces - len(file_type)) + '  |\n' \
                        + '|  ' + file_type_description1 + ' ' * (num_of_spaces - len(file_type_description1)) + '  |\n' \
                        + '|  ' + output + ' ' * (num_of_spaces - len(output)) + '  |\n' \
                        + '|  ' + output_description1 + ' ' * (num_of_spaces - len(output_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'mergeBranch':
                    function = '\033[1;37;40msctreeshap.sctreeshap.mergeBranch\033[0m'
                    api = 'sctreeshap.sctreeshap.mergeBranch(data=None, branch_name=None)'
                    description =                    'Description: merge all clusters under a given branch.'
                    data =                          'Parameters:  data: AnnData or DataFrame'
                    branch_name =                   '             branch_name: str'
                    branch_name_description1 =      '             |  The clusters under the branch will merge, relabelled as the branch_name.'
                    return_description =            'Return:      DataFrame or AnnData.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                        + '|  ' + branch_name + ' ' * (num_of_spaces - len(branch_name)) + '  |\n' \
                        + '|  ' + branch_name_description1 + ' ' * (num_of_spaces - len(branch_name_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'AnnData_to_DataFrame':
                    function = '\033[1;37;40msctreeshap.sctreeshap.AnnData_to_DataFrame\033[0m'
                    api = 'sctreeshap.sctreeshap.AnnData_to_DataFrame(adata)'
                    description =                    'Description: convert AnnData to DataFrame.'
                    adata =                          'Parameters:  adata: AnnData'
                    adata_description1 =             '             |  An AnnData object in anndata package.'
                    return_description =             'Return:      DataFrame.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + adata + ' ' * (num_of_spaces - len(adata)) + '  |\n' \
                        + '|  ' + adata_description1 + ' ' * (num_of_spaces - len(adata_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'DataFrame_to_AnnData':
                    function = '\033[1;37;40msctreeshap.sctreeshap.DataFrame_to_AnnData\033[0m'
                    api = 'sctreeshap.sctreeshap.DataFrame_to_AnnData(data)'
                    description =                    'Description: convert DataFrame to AnnData.'
                    data =                          'Parameters:  data: DataFrame'
                    data_description1 =             '             |  A DataFrame object in pandas package.'
                    return_description =             'Return:      AnnData.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                        + '|  ' + data_description1 + ' ' * (num_of_spaces - len(data_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'geneFiltering':
                    function = '\033[1;37;40msctreeshap.sctreeshap.geneFiltering\033[0m'
                    api = 'sctreeshap.sctreeshap.geneFiltering(data=None, min_partial=None, gene_set=None, gene_prefix=None)'
                    description =                    'Description: filter genes customly.'
                    data =                          'Parameters:  data: DataFrame or AnnData'
                    min_partial =                   '             min_partial: float'
                    min_partial_description1 =      '             |  If not None, filter genes expressed in less than min_partial * 100% cells.'
                    gene_set =                      '             gene_set: list or tuple'
                    gene_set_description1 =         '             |  A list or a tuple of genes to be filtered.'
                    gene_prefix =                   '             gene_prefix: list or tuple'
                    gene_prefix_description1 =      '             |  Genes with prefix in gene_prefix will be filtered.'
                    return_description =             'Return:      AnnData or DataFrame.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                        + '|  ' + min_partial + ' ' * (num_of_spaces - len(min_partial)) + '  |\n' \
                        + '|  ' + min_partial_description1 + ' ' * (num_of_spaces - len(min_partial_description1)) + '  |\n' \
                        + '|  ' + gene_set + ' ' * (num_of_spaces - len(gene_set)) + '  |\n' \
                        + '|  ' + gene_set_description1 + ' ' * (num_of_spaces - len(gene_set_description1)) + '  |\n' \
                        + '|  ' + gene_prefix + ' ' * (num_of_spaces - len(gene_prefix)) + '  |\n' \
                        + '|  ' + gene_prefix_description1 + ' ' * (num_of_spaces - len(gene_prefix_description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'explainBinary':
                    function = '\033[1;37;40msctreeshap.sctreeshap.explainBinary\033[0m'
                    api = 'sctreeshap.sctreeshap.explainBinary(data=None, cluster_name=None, use_SMOTE=False, nthread=32, '
                    api1 = 'shap_params=None)'
                    description =                    'Description: do binary classification and generate shap figures.'
                    data =                          'Parameters:  data: DataFrame or AnnData'
                    cluster_name =                  '             cluster_name: str'
                    cluster_name_description1 =     '             |  The target cluster for classification.'
                    use_SMOTE =                     '             use_SMOTE: bool'
                    use_SMOTE_description1 =        '             |  True if you want to use SMOTE to resample.'
                    nthread =                       '             nthread: int'
                    nthread_description1 =          '             |  The number of running threads.'
                    shap_params =                   '             shap_params: dictionary'
                    shap_params_description1 =      '             |  Keys: ["max_display", "bar_plot", "beeswarm", "force_plot", "heat_map", "decision_plot"]'
                    shap_params_description2 =      '             |  Values: "max_display" reflects an int, which determines the maximum number of genes to display'
                    shap_params_description3 =      '             |  in shap figures, while each other key is reflected to a bool which indicates whether to '
                    shap_params_description4 =      '             |  output this kind of shap figures.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + api1 + ' ' * (num_of_spaces - len(api1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                        + '|  ' + cluster_name + ' ' * (num_of_spaces - len(cluster_name)) + '  |\n' \
                        + '|  ' + cluster_name_description1 + ' ' * (num_of_spaces - len(cluster_name_description1)) + '  |\n' \
                        + '|  ' + use_SMOTE + ' ' * (num_of_spaces - len(use_SMOTE)) + '  |\n' \
                        + '|  ' + use_SMOTE_description1 + ' ' * (num_of_spaces - len(use_SMOTE_description1)) + '  |\n' \
                        + '|  ' + nthread + ' ' * (num_of_spaces - len(nthread)) + '  |\n' \
                        + '|  ' + nthread_description1 + ' ' * (num_of_spaces - len(nthread_description1)) + '  |\n' \
                        + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                        + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                        + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                        + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                        + '|  ' + shap_params_description4 + ' ' * (num_of_spaces - len(shap_params_description4)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'explainMulti':
                    function = '\033[1;37;40msctreeshap.sctreeshap.explainMulti\033[0m'
                    api = 'sctreeshap.sctreeshap.explainMulti(data=None, use_SMOTE=False, nthread=32, shap_params=None)'
                    description =                    'Description: do multi-classification and generate shap figures.'
                    data =                          'Parameters:  data: DataFrame or AnnData'
                    use_SMOTE =                     '             use_SMOTE: bool'
                    use_SMOTE_description1 =        '             |  True if you want to use SMOTE to resample.'
                    nthread =                       '             nthread: int'
                    nthread_description1 =          '             |  The number of running threads.'
                    shap_params =                   '             shap_params: dictionary'
                    shap_params_description1 =      '             |  Keys: ["max_display", "bar_plot", "beeswarm", "decision_plot"]'
                    shap_params_description2 =      '             |  Values: "max_display" reflects an int, which determines the maximum number of genes to display'
                    shap_params_description3 =      '             |  in shap figures, while each other key is reflected to a bool which indicates whether to '
                    shap_params_description4 =      '             |  output this kind of shap figures.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + data + ' ' * (num_of_spaces - len(data)) + '  |\n' \
                        + '|  ' + use_SMOTE + ' ' * (num_of_spaces - len(use_SMOTE)) + '  |\n' \
                        + '|  ' + use_SMOTE_description1 + ' ' * (num_of_spaces - len(use_SMOTE_description1)) + '  |\n' \
                        + '|  ' + nthread + ' ' * (num_of_spaces - len(nthread)) + '  |\n' \
                        + '|  ' + nthread_description1 + ' ' * (num_of_spaces - len(nthread_description1)) + '  |\n' \
                        + '|  ' + shap_params + ' ' * (num_of_spaces - len(shap_params)) + '  |\n' \
                        + '|  ' + shap_params_description1 + ' ' * (num_of_spaces - len(shap_params_description1)) + '  |\n' \
                        + '|  ' + shap_params_description2 + ' ' * (num_of_spaces - len(shap_params_description2)) + '  |\n' \
                        + '|  ' + shap_params_description3 + ' ' * (num_of_spaces - len(shap_params_description3)) + '  |\n' \
                        + '|  ' + shap_params_description4 + ' ' * (num_of_spaces - len(shap_params_description4)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'getShapValues':
                    function = '\033[1;37;40msctreeshap.sctreeshap.getShapValues\033[0m'
                    api = 'sctreeshap.sctreeshap.getShapValues()'
                    description =                    "Description: get shap values of the last job (available after 'a.explainBinary()' or 'a.explainMulti()')."
                    return_description =             'Return:      ndarray.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                elif cmd == 'getTopGenes':
                    function = '\033[1;37;40msctreeshap.sctreeshap.getTopGenes\033[0m'
                    api = 'sctreeshap.sctreeshap.getTopGenes(max_display=None, shap_values=None, feature_names=None)'
                    description =                    'Description: get top genes of max absolute mean shap values. (You can just simply call it without any '
                    description1 =                   '             parameters after explainBinary() or explainMulti()).'
                    max_display =                    'Parameters:  max_display: int'
                    max_display_description1 =       '             |  The the number of top genes you want to derive.'
                    shap_values =                    '             shap_values: list or ndarray'
                    shap_values_description1 =       '             |  This can be derived from getShapValues(), defaultly from last explainBinary()/explainMulti().'
                    feature_names =                  '             feature_names: ndarray, list or tuple'
                    feature_names_description1 =     '             |  The gene set (columns of cell-gene matrix), must correspond to order of shap_values, defaultly'
                    feature_names_description2 =     '             |  from last explainBinary()/explainMulti().'
                    return_description =             'Return:      ndarray, an ordered set of top genes of maximum absolute mean shap values.'
                    print( ' __' + '_' * num_of_spaces + '__ \n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + function + ' ' * (num_of_spaces - len(function) + 14) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + api + ' ' * (num_of_spaces - len(api)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + description + ' ' * (num_of_spaces - len(description)) + '  |\n' \
                        + '|  ' + description1 + ' ' * (num_of_spaces - len(description1)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + max_display + ' ' * (num_of_spaces - len(max_display)) + '  |\n' \
                        + '|  ' + max_display_description1 + ' ' * (num_of_spaces - len(max_display_description1)) + '  |\n' \
                        + '|  ' + shap_values + ' ' * (num_of_spaces - len(shap_values)) + '  |\n' \
                        + '|  ' + shap_values_description1 + ' ' * (num_of_spaces - len(shap_values_description1)) + '  |\n' \
                        + '|  ' + feature_names + ' ' * (num_of_spaces - len(feature_names)) + '  |\n' \
                        + '|  ' + feature_names_description1 + ' ' * (num_of_spaces - len(feature_names_description1)) + '  |\n' \
                        + '|  ' + feature_names_description2 + ' ' * (num_of_spaces - len(feature_names_description2)) + '  |\n' \
                        + '|  ' + emptyline + ' ' * (num_of_spaces - len(emptyline)) + '  |\n' \
                        + '|  ' + return_description + ' ' * (num_of_spaces - len(return_description)) + '  |\n' \
                        + '|__' + '_' * num_of_spaces + '__|')
                else:
                    print("Unrecognized item:", cmd)
            else:
                print("Unrecognized command:", cmd[:4])
