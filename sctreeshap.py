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

# 1. Construct a sctreeshap object with a given cluster tree:
#   a = sctreeshap(tree_arr=None)
# where tree_arr is a dict.
#           Let n <- len(tree_arr);
#                   then n represents the number of non-leaf nodes in the cluster tree;
#               tree_arr[str] represents the node of name str in the cluster tree;
#                   tree_arr[str] can be a list or a tuple of strings, representing the name of childs of the node (from left to right);
#                   e.g. tree_arr['n1'] = ('n2', 'n70') represents a node named 'n1', whose left child is 'n2' and right child is 'n70';
#               note that you do not need to create nodes for clusters, since they are leaf nodes and have no childs.
# Or:
#   a = sctreeshap(tree_arr)
# where tree_arr is a dict.
#            Let n <- len(tree_arr);
#                   then n represents the number of nodes (root excluded) in the cluster tree;
#               tree_arr[str] represents the node of name str in the cluster tree;
#                   tree_arr[str] should be a list or a tuple of a string and an int, representing the name of parent of the node and which child it is (from left to right, start from 0);
#                   e.g. tree_arr['n2'] = ('n1', 0) represents a node named 'n2', who is the leftmost child of 'n1';
#               note that you do not need to create a node for the root, since it does not have a parent.
# If tree_arr == None, then it does not construct a cluster tree,
# Return: an sctreeshap object.
#
# 2. Set default data directory:
#   a.setDataDirectory(data_directory)
# where data_directory is a string representing the directory of the default input file.
# Note: if you want to clear default settings, run "a.set...(None)".
# Return: None
#
# 3. Set default data file type:
#   a.setFileType(filetype)
# where filetype can be 'csv' or 'pkl', representing the input file type.
# Return: None
#
# 4. Set default dataset:
#   a.setDataSet(data)
# where data is a DataFrame or AnnData, representing the default dataset, which would be prior to data directory.
# Note: the priority order is "dataset parameter -> data directory parameter -> default dataset -> default data directory".
# Return: None
#
# 5. Set default branch:
#   a.setBranch(branch_name)
# where branch_name is a string representing the branch's name, which would be defaultedly chosen.
# Return: None
#
# 6. Set default cluster:
#   a.setCluster(cluster_name)
# where cluster_name is a string representing the cluster's name, which would be defaultedly chosen.
# Return: None
#
# 7. Set default cluster set:
#   a.setClusterSet(cluster_set)
# where cluster_set is a list or tuple of strings containing all target clusters to choose.
# Return: None
#
# 8. Set default shap plots parameters of explainBinary:
#   a.setShapParamsBinary(shap_params)
# where shap_params is a dict, including five keys: ["bar_plot", "beeswarm", "force_plot", "heat_map", "decision_plot"], which is defaultedly set as [True, True, False, False, False];
#           you can reset the dict to determine what kinds of figures to output.
# Return: None
#
# 9. Set default shap plots parameters of explainMulti:
#   a.setShapParamsMulti(shap_params)
# where shap_params is a dict, including three keys: ["bar_plot", "beeswarm", "decision_plot"], which is defaultedly set as [True, False, False];
#           you can reset the dict to determine what kinds of figures to output.
# Return: None
#
# 10. Get XGBClassifier of the last job: (available after each 'a.explainBinary()' or 'a.explainMulti()')
#   a.getClassifier()
# Return: <class 'xgboost.sklearn.XGBClassifier'> object
#
# 11. Get shap explainer of the last job: (available after each 'a.explainBinary()' or 'a.explainMulti()')
#   a.getExplainer()
# Return: <class 'shap.explainers._tree.Tree'> object
#
# 12. Get shap values of the last job: (available after each 'a.explainBinary()' or 'a.explainMulti()')
#   a.getShapValues()
# Return: an ndarray.
#
# 13. Find which branch a specific cluster is in: 
#   a.find(cluster_name=None)
# where cluster_name is a string representing the cluster's name, e.g. "Exc L5-6 THEMIS FGF10".
#           if cluster_name == None: choose default cluster.
# Return: a string representing the path.
#
# 14. List the genes of a branch:
#   a.list(branch_name=None)
# where branch_name is a string representing the branch's name, e.g. "n48".
#           if branch_name == None: choose default branch; if default is still None, list all clusters.
# Return: a list including all cluster names under the branch.
#
# 15. Read cells from a branch:
#   a.readData(data_directory=None, filetype='csv', branch_name=None, cluster_set=[], use_cluster_set=False, output='DataFrame')
# where data_directory is a string representing the directory of the file, e.g. "~/xhx/python/neuron_full.pkl";
#           if data_directory == None: use default data directory.
#       filetype is the file type, default is 'csv', which can be reset to 'pkl';
#       branch_name is a string representing the target branch, e.g. "n48";
#           if branch_name == None: choose default branch; if default is still None, read the whole dataset.
#       cluster_set is a list or tuple of strings containing all target clusters to choose;
#       use_cluster_set indicates whether to activate choose from cluster_set;
#       output can be 'DataFrame' or 'AnnData', which indicates return type.
# Return: a DataFrame or AnnData object.
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
            if typeOfTree == None:
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
                    if self.__root == None:
                        self.__root = key
                    else:
                        print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') receives an invalid dict (not a tree structure).")
                        return False
            if self.__root == None:
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

    def __init__(self, tree_arr=None):
        self.numOfClusters = 0
        self.__clusterDict = {}
        self.__dataDirectory = None
        self.__fileType = 'csv'
        self.__dataSet = None
        self.__branch = None
        self.__cluster = None
        self.__clusterSet = []
        self.__waitingMessage = None
        self.__isFinished = False
        self.__XGBClassifer = None
        self.__explainer = None
        self.__shapValues = None
        self.__TreeNode = None
        self.__parent = {}
        self.__root = None
        self.__visited = []
        self.__shapParamsBinary = {
            "bar_plot": True,
            "beeswarm": True,
            "force_plot": False,
            "heat_map": False,
            "decision_plot": False
        }
        self.__shapParamsMulti = {
            "bar_plot": True,
            "beeswarm": False,
            "decision_plot": False
        }
        if not self.__checkClusterTree(tree_arr):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.sctreeshap()' (in file '" + __file__ + "') throws an exception.")
            return None
    
    def setDataDirectory(self, data_directory):
        if data_directory is None:
            self.__dataDirectory = None
            return None
        if not isinstance(data_directory, str):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setDataDirectory()' (in file '" + __file__ + "') receives an invalid data_directory of wrong type.")
            return -1
        self.__dataDirectory = data_directory
        return None
    
    def setFileType(self, filetype):
        if filetype is None:
            self.__fileType = 'csv'
            return None
        if filetype != 'csv' and filetype != 'pkl':
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setFileType()' (in file '" + __file__ + "') receives an invalid filetype parameter (must be 'csv' or 'pkl').")
            return -1
        self.__fileType = filetype
        return None
    
    def setDataSet(self, data):
        if data is None:
            self.__dataSet = None
            return None
        if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, ad._core.anndata.AnnData):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setDataSet()' (in file '" + __file__ + "') receives an invalid dataset of wrong type (must be 'AnnData' or 'DataFrame').")
            return -1
        self.__dataSet = data
        return None
    
    def setBranch(self, branch_name):
        if branch_name is None:
            self.__branch = None
            return None
        if not isinstance(branch_name, str):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setBranch()' (in file '" + __file__ + "') receives an invalid branch_name of wrong type.")
            return -1
        self.__branch = branch_name
        return None
    
    def setCluster(self, cluster_name):
        if cluster_name is None:
            self.__cluster = None
            return None
        if not isinstance(cluster_name, str):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setCluster()' (in file '" + __file__ + "') receives an invalid cluster_name of wrong type.")
            return -1
        self.__cluster = cluster_name
        return None
    
    def setClusterSet(self, cluster_set):
        if cluster_set is None:
            self.__clusterSet = None
            return None
        if not isinstance(cluster_set, list) and not isinstance(cluster_set, tuple):
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.setClusterSet()' (in file '" + __file__ + "') receives an invalid cluster_set of wrong type.")
            return -1
        self.__clusterSet = cluster_set
        return None
    
    def setShapParamsBinary(self, shap_params):
        if shap_params is None:
            self.__shapParamsBinary = {
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
    
    def setShapParamsMulti(self, shap_params):
        if shap_params is None:
            self.__shapParamsMulti = {
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

    def getClassifier(self):
        return self.__XGBClassifer

    def getExplainer(self):
        return self.__explainer
    
    def getShapValues(self):
        return self.__shapValues

    def find(self, cluster_name=None, root=None, path="ROOT"):
        if root == None:
            root = self.__root
        if cluster_name == None:
            cluster_name = self.__cluster
            if cluster_name == None:
                self.__isFinished = True
                time.sleep(0.2)
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.find()' (in file '" + __file__ + "') requires a target cluster name.")
                return -1
        if root not in self.__TreeNode.keys():
            return "Cluster " + cluster_name + " not found!"
        childs = self.__TreeNode[root]
        for item in childs:
            if item == cluster_name:
                return path + " --> " + root + " --> " + cluster_name
            else:
                result = self.find(cluster_name, item, path + " --> " + root)
                if result != "Cluster " + cluster_name + " not found!":
                    return result
        return "Cluster " + cluster_name + " not found!"
        
    def list(self, branch_name=None):
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
            result = result + self.list(item)
        return result

    def readData(self, data_directory=None, filetype=None, branch_name=None, cluster_set=[], use_cluster_set=False, output='DataFrame'):
        if data_directory == None:
            data_directory = self.__dataDirectory
        if not use_cluster_set and branch_name == None:
            branch_name = self.__branch
        data = None
        if filetype is None:
            filetype = self.__fileType
        if filetype == 'csv':
            try:
                data = pd.read_csv(data_directory)
            except:
                self.__isFinished = True
                time.sleep(0.2)
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        elif filetype == 'pkl':
            try:
                data = pd.read_pickle(data_directory)
            except:
                self.__isFinished = True
                time.sleep(0.2)
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        else:
            self.__isFinished = True
            time.sleep(0.2)
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') receives an unrecognized file type.")
            return -1
        if use_cluster_set:
            if (not isinstance(cluster_set, list) and not isinstance(cluster_set, tuple)) or len(cluster_set) == 0:
                cluster_set = self.__clusterSet
            return data[data['cluster'].isin(cluster_set)]
        if branch_name != None:
            clusters = self.list(branch_name)
            if clusters == -1:
                time.sleep(0.2)
                print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') throws an exception: '" + data_directory + "' no such file or directory.")
                return -1
        if output == "DataFrame":
            return data
        elif output == "AnnData":
            return self.DataFrame_to_AnnData(data)
        else:
            print("\033[1;31;40mError:\033[0m method 'sctreeshap.readData()' (in file '" + __file__ + "') receives a wrong output format parameter (must be 'AnnData' or 'DataFrame').")
            return -1

    def AnnData_to_DataFrame(self, adata):
        return pd.concat([pd.DataFrame(adata.X, columns=adata.var.index.values).reset_index(drop=True), adata.obs.reset_index(drop=True)], axis=1, join="inner")

    def DataFrame_to_AnnData(self, data):
        obs = pd.DataFrame(data["cluster"], columns=["cluster"])
        obs["cluster"] = obs.cluster.astype("category")
        data.drop(["cluster", "Unnamed: 0"], axis=1, inplace=True)
        var = pd.DataFrame(index=data.columns.values)
        X = np.array(data)
        return ad.AnnData(np.array(data), obs=obs, var=var, dtype="float")

    def explainBinary(self, data=None, cluster_name=None, use_SMOTE=False, shap_output_directory=None, nthread=32, shap_params=None):
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

        if shap_output_directory != None:
            # Generating shap values file
            self.__waitingMessage = "Writing shap values to '" + shap_output_directory + "'.."
            self.__isFinished = False
            thread_writeShapValues = threading.Thread(target=showProcess)
            thread_writeShapValues.start()
            shap_values_file = pd.DataFrame(self.__shapValues, index=list(data.index.values))
            shap_values_file.columns = list(x.columns.values)
            shap_values_file.to_csv(shap_output_directory)
            self.__isFinished = True
            thread_writeShapValues.join()
        else:
            print("No directory detected. Skipped shap values output.")

        # Generating shap figures
        print("Generating shap figures..")
        if shap_params == None:
            shap_params = self.__shapParamsBinary
        if "bar_plot" not in shap_params or shap_params["bar_plot"]:
            print("     Drawing bar plot..")
            plt.figure(1)
            plt.title("Target Cluster: " + cluster_name)
            shap.summary_plot(self.__shapValues, x_test, feature_names=x.columns.values, max_display=10, plot_type='bar', show=False)
            plt.show()
        if "beeswarm" not in shap_params or shap_params["beeswarm"]:
            print("     Drawing beeswarm plot..")
            plt.figure(2)
            plt.title("Target Cluster: " + cluster_name)
            shap.summary_plot(self.__shapValues, x_test, feature_names=x.columns.values, max_display=10)
            plt.show()
        if "force_plot" not in shap_params or shap_params["force_plot"]:
            print("     Drawing force plot..")
            print("     \033[1;33;40mWarning:\033[0m: force plot has not been stably supported yet.")
            shap.initjs()
            shap.plots.force(self.__explainer.expected_value, self.__shapValues, x_test, feature_names=x.columns.values, show=False)
        if "heat_map" not in shap_params or shap_params["heat_map"]:
            print("     Drawing heat map..")
            plt.figure(3)
            plt.title("Target Cluster: " + cluster_name)
            shap.plots.heatmap(self.__explainer(x_test), show=False)
            plt.show()
        if "decision_plot" not in shap_params or shap_params["decision_plot"]:
            print("     Drawing decision plot..")
            plt.figure(4)
            plt.title("Target Cluster: " + cluster_name)
            y_pred = pd.DataFrame(y_pred).to_np()
            x_target = x_test[y_pred == 1]
            shap_values = self.__explainer.shap_values(x_target, approximate=True)
            shap.decision_plot(self.__explainer.expected_value, shap_values, x_target, link='logit', show=False)
            plt.show()

    def explainMulti(self, data=None, use_SMOTE=False, shap_output_directory=None, nthread=32, shap_params=None):
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
        self.__XGBClassifer = XGBClassifier(objective="multi:softmax", nthread=nthread, eval_metric="mlogloss", random_state=42, use_label_encoder=False)
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

        if shap_output_directory != None:
            # Generating shap values file
            self.__waitingMessage = "Writing shap values to '" + shap_output_directory + "'.."
            self.__isFinished = False
            thread_writeShapValues = threading.Thread(target=showProcess)
            thread_writeShapValues.start()
            shap_values_file = pd.DataFrame(self.__shapValues, index=list(data.index.values))
            shap_values_file.columns = list(x.columns.values)
            shap_values_file.to_csv(shap_output_directory)
            self.__isFinished = True
            thread_writeShapValues.join()
        else:
            print("No directory detected. Skipped shap values output.")

        # Generating shap figures
        print("Generating shap figures..")
        if shap_params == None:
            shap_params = self.__shapParamsMulti
        if "bar_plot" not in shap_params or shap_params["bar_plot"]:
            print("     Drawing bar plot..")
            plt.figure(1)
            shap.summary_plot(self.__shapValues, x_test, feature_names=x.columns.values, max_display=10, show=False)
            plt.show()
        if "beeswarm" not in shap_params or shap_params["beeswarm"]:
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
                shap.summary_plot(self.__shapValues[self.clusterDict[key]], x_test, feature_names=x.columns.values, max_display=10, show=False)
                index += 1
            figure.subplots_adjust(right=5, top=rows*3.5, hspace=0.2, wspace=0.2)
            plt.show()
        if "decision_plot" not in shap_params or shap_params["decision_plot"]:
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
                y_pred_i = y_pred[y_pred.columns[self.clusterDict[key]]].to_np()
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
    
    def __str__(self):
        default_data_directory = "Default data directory: "
        if self.__dataDirectory != None:
            default_data_directory += self.__dataDirectory
        else:
            default_data_directory += 'None'
        default_selected_branch = "Default selected branch: "
        if self.__branch != None:
            default_selected_branch += self.__branch
        else:
            default_selected_branch += 'None'
        default_target_cluster = "Default target cluster: "
        if self.__cluster != None:
            default_target_cluster += self.__cluster
        else:
            default_target_cluster += 'None'
        num_of_spaces = max(len(default_data_directory), len(default_selected_branch), len(default_target_cluster))
        return ' __' + '_' * num_of_spaces + '__ \n' \
            + '|  ' + ' ' * num_of_spaces + '  |\n' \
            + '|  ' + default_data_directory + ' ' * (num_of_spaces - len(default_data_directory)) + '  |\n' \
            + '|  ' + default_selected_branch + ' ' * (num_of_spaces - len(default_selected_branch)) + '  |\n' \
            + '|  ' + default_target_cluster + ' ' * (num_of_spaces - len(default_target_cluster)) + '  |\n' \
            + '|__' + '_' * num_of_spaces + '__|'
