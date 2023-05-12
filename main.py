import random
import sys
import time
import joblib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import *
from sklearn.metrics import silhouette_score,silhouette_samples
def choose(exp_ver=2,k_folder_option=False):
    if exp_ver == 2:
        if k_folder_option:
            help(process(5, True, 200, True, 0.8, 0.0, 0.2, True, 1, True, True, False, True, 'model.pkl', 'linear', get_help=False))  # 5折交叉验证,实验2
        else:
            help(process(1, False, 200, True, 0.8, 0.0, 0.2, True, 1, True, True, False,True, 'model.pkl', 'linear', get_help=False))  # 普通训练，实验2
    elif exp_ver == 3:
        if k_folder_option:
            help(process(5, True, 200, True, 0.8, 0.0, 0.2, True, 1, True, True, False, True, 'model.pkl', 'linear', get_help=False, experiment_ver=3, n_clusters=3))  # 5折交叉验证，实验3
        else:
            help(process(1, False, 200, True, 0.8, 0.0, 0.2, True, 1, True, True, False, True, 'model.pkl', 'linear', get_help=False, experiment_ver=3, n_clusters=3))  # 普通训练，实验3
    else:
        raise ValueError('can not found exp_ver=', exp_ver, 'k_folder_option=', k_folder_option)




class process():
    # load iris dataset return model
    def __init__(self, k_folder=1, k_folder_test=False, enhancement_level=0, data_enhancement=False, size_train=0.6,
                 size_val=0.2, size_test=0.2, random_set=True, random_seed=1, draw_fusion_matrix=False, save_P_R=False,
                 runtime_broker=False, save_model=True, model_name='model.pkl', kernel='linear', get_help=False,
                 experiment_ver=2, n_clusters=3):
        """
        process.__init__() function:
            usage:
                k_folder :
                    if k_folder_test == False ,this option is unavailable,else use cross validation(k-folder),default = 1
                k_folder_test :
                    if this option == True , we will use cross validation(k_folder),else,we will use common training way,default = False
                enhancement_level :
                    if data_enhancement == False ,this option is unavailable, else we will add noise on each data,default = 0
                data_enhancement :
                    if this option == False, we will never add noice on data,however this option may cause accuracy reducing,else we will add some noise to the data
                size_train :
                    if k_folder_test == True, this option is unavailable,else we will split dataset -> train_data(length = int(len(dataset)*size_train)),default = 0.6
                size_val :
                    if k_folder_test == True, this option is unavailable, else we will split dataset -> val_data(length = int(len(dataset)*size_val)),default = 0.2
                size_test :
                    if k_folder_test == True, this option is unavailable, else we will split dataset -> test_data(length = int(len(dataset)*size_test)),default = 0.2
                random_set :
                    if random_set == True, we will disrupt dataset according to the random_seed, else we will never shuffle the dataset, default = True
                random_seed :
                    if random_set == False, this option is unavailable, else we will disrupt dataset according to the random_seed, default = 1
                draw_fusion_matrix :
                    if draw_fusion_matrix == True, we will draw confusion matrix,else we will never draw confusion matrix, default = False
                save_P_R :
                    if save_P_R == True, we will save P-R curve by matplotlib.pyplot ,else we will never save P_R curve to your computer,default = False
                runtime_broker :
                    if runtime_broker == True, we will stop when the photograph is OK,else we will save the photograph directly,default = False
                save_model :
                    if save_model == True, we will dump model to your computer (current path) by joblib,else we will not dump model,default = True
                model_name :
                    if save_model == False, this option is unavailable,else we will use this name to save model, default = 'model.pkl'
                kernel :
                    SupportVectorMachine Kernel function -> default = 'linear'
                help :
                    get_help,default = False
                experiment_ver :
                    experiment version you can choose 2 or 3 to train with different type of model (exp_2 : SupportVectorMechine,exp_3 : K-NearestNeighbor)
                n_neighbor :
                    define n_neighbor(from sklearn.neighbor.KNeighborsClassifier(n_neighbor)) only experiment_ver == 3,this choice is available
        """
        self.get_help = get_help
        if get_help:
            self.help()
        else:
            self.enhancement_level = enhancement_level
            self.data_enhancement = data_enhancement
            self.random_set = random_set
            self.seed = random_seed
            self.k_folder = k_folder
            self.k_folder_test = k_folder_test
            self.size_train = size_train
            self.size_val = size_val
            self.size_test = size_test
            self.data, self.label = [], []
            self.y_pred = []
            self.y_pred0 = []
            self.model = None
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = [], [], [], [], [], []
            self.draw = draw_fusion_matrix
            self.draw_PR = save_P_R
            self.broker = runtime_broker
            self.save_model = save_model
            self.model_name = model_name
            self.kernel = kernel
            self.exp_version = experiment_ver
            self.n_neighbors = n_clusters
            if sys.version_info >= (3, 10):
                match experiment_ver:
                    case 2:
                        self._forward_ver_2()
                    case 3:
                        self._forward_ver_3()
                    case _:
                        raise ValueError('can not found this version')
            elif sys.version_info >= (3, 0):
                if experiment_ver == 2:
                    self._forward_ver_2()
                elif experiment_ver == 3:
                    self._forward_ver_3()
                else:
                    raise ValueError('can not found this version')
            else:
                raise ValueError('can not run this script in current version')
            
    def __copyright__(self):
        """this python project is edited by LRY, all rights reserved (c) 2020~2023
        """

    def help(self):
        """
        help function in Process class
        usage:
            1. help(process())
            2. process(get_help = True)
        """
        print('Class usage')
        print(self.__init__.__doc__)

    def _make_dataset_ver_2(self):
        if not self.data_enhancement:
            iris = load_iris()
            X = iris.data
            Y = iris.target
            X = X[Y < 2, :2]
            Y = Y[Y < 2]
            # 标准化
            std = StandardScaler()
            std.fit(X)
            X_standard = std.transform(X)
            return X_standard, Y
        else:
            iris = load_iris()
            X = iris.data
            Y = iris.target
            X = X[Y < 2, :2]
            Y = Y[Y < 2]
            # enhance here
            assert type(self.enhancement_level) == int
            assert len(X) == len(Y)
            length = len(X)
            for i in range(self.enhancement_level):
                for j in range(length):
                    X_0 = X[j]
                    X_00 = X_0[0]
                    X_01 = X_0[1]
                    ran_0 = 0.5 / (self.enhancement_level + 1)
                    X = np.append(X, (
                        X_00 + random.uniform(-self.enhancement_level * ran_0, self.enhancement_level * ran_0),
                        X_01 + random.uniform(-self.enhancement_level * ran_0, self.enhancement_level * ran_0)))
                    Y = np.append(Y, Y[j])
                    X = X.reshape(-1, 2)
            # 标准化
            std = StandardScaler()
            std.fit(X)
            X_standard = std.transform(X)
            return X_standard, Y

    def _dump_model(self, name_change=False, new_name=''):
        assert self.save_model == True
        assert type(self.model_name) == str
        if name_change:
            joblib.dump(self.model, new_name)
        else:
            joblib.dump(self.model, self.model_name)

    def _train(self, X, Y):
        kMeans = None
        svc = None
        if self.exp_version == 2:
            svc = SVC(kernel=self.kernel)
            svc.fit(X, Y)
            self.model = svc
        elif self.exp_version == 3:
            kMeans = KMeans(n_clusters=self.n_neighbors,init='k-means++',n_init=100)
            kMeans.fit(X)
            self.model = kMeans
        return svc if svc is not None else kMeans

    def _model_predict(self, X):
        if self.model == None:
            raise ValueError('the model have not train yet')
        return self.model.predict(X)

    def _draw_P_R(self, y_true, y_pred, change_name=False, new_name=''):
        if self.exp_version == 3:
            RuntimeWarning('P-R is not support in version 3')
            return
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        plt.figure("P-R Curve")
        plt.title('Precision/Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision)
        if self.broker:
            plt.show()
        if self.draw_PR:
            if change_name:
                plt.savefig(new_name)
            else:
                plt.savefig('P-R_curve.png')
        plt.close()

    def _draw_confusion_matrix(self, y_true, y_pred, change_name=False, new_name=''):
        assert len(y_pred) == len(y_true)
        if self.draw:
            if change_name:
                self._draw_plt_confusion_matrix(y_true, y_pred, add_acc=True, title=new_name)
            else:
                self._draw_plt_confusion_matrix(y_true, y_pred, add_acc=True)
        else:
            print(confusion_matrix(y_true, y_pred))

    def _draw_plt_confusion_matrix(self, y_true, y_pred, num_classes=2, classes=['0', '1'], title='confusion_matrix',
                                   add_acc=False):
        assert len(y_true) == len(y_pred)
        assert self.draw == True
        assert len(classes) == num_classes
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        acc = 0
        total = 0
        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1
            if y_pred[i] == y_true[i]:
                acc += 1
                total += 1
            else:
                total += 1
        # draw
        print(matrix)
        plt.matshow(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                plt.text(x=j, y=i, s=matrix[i, j])
        plt.xlabel('pred_label')
        plt.ylabel('true_label')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        if add_acc:
            plt.title(title + ",acc=" + str(acc / total))
        else:
            plt.title(title)
        if self.broker:
            plt.show()
        if add_acc:
            plt.savefig(title + ",acc=" + str(acc / total) + ".png")
        else:
            plt.savefig(title + ".png")

    def _split_dataset(self):
        size_train = self.size_train
        size_val = self.size_val
        size_test = self.size_test
        assert size_train + size_val + size_test == 1.0
        X = self.data
        Y = self.label
        train_X = X[0:int(size_train * len(X))]
        train_Y = Y[0:int(size_train * len(X))]
        val_X = X[int(size_train * len(X)):int((len(X) - len(X) * size_test))]
        val_Y = Y[int(size_train * len(X)):int((len(X) - len(X) * size_test))]
        test_X = X[int((len(X) - len(X) * size_test)):len(X)]
        test_Y = Y[int((len(X) - len(X) * size_test)):len(X)]
        return train_X, train_Y, val_X, val_Y, test_X, test_Y

    def _random_setting(self):
        assert len(self.data) == len(self.label)
        random.seed(self.seed)
        if not self.random_set:
            return self.data, self.label
        else:
            length = len(self.data)
            L1 = random.sample(range(0, length), length)
            X0 = []
            Y0 = []
            for i in range(len(L1)):
                X0.append(self.data[L1[i]])
                Y0.append(self.label[L1[i]])
            return X0, Y0

    def _forward_ver_2(self):
        self.data, self.label = self._make_dataset_ver_2()
        if self.random_set:
            self.data, self.label = self._random_setting()
        if self.k_folder_test:
            return self._forward_with_k_folder_ver_2()
        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = self._split_dataset()
        return self._forward_none_k_folder_ver_2()

    def _forward_none_k_folder_ver_2(self):
        train_x, train_y, val_x, val_y, test_x, test_y = self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y
        self.model = self._train(train_x, train_y)
        if self.save_model:
            self._dump_model()
        self.y_pred = self._model_predict(test_x)
        y_pred = self.y_pred
        y_pred = np.array(y_pred)
        target_name = ['0', '1']
        print(classification_report(test_y, y_pred, target_names=target_name))
        self._draw_P_R(test_y, y_pred)
        print('混淆矩阵为：')
        self._draw_confusion_matrix(test_y, y_pred)

    def _split_k_folder(self, current_folder, total_folder, X, Y):
        """
        input
            current_folder : 0~self.k_folder-1
            total_folder : self.k_folder
            X : data
            Y : label
        return
            train_X, train_Y, test_X, test_Y
        """
        assert type(self.k_folder) == int
        assert self.k_folder_test == True
        assert self.k_folder <= int(len(self.data) / 2)
        assert self.k_folder > current_folder
        test_X = X[int(current_folder * len(X) / total_folder):int((current_folder + 1) * len(X) / total_folder)]
        test_Y = Y[int(current_folder * len(X) / total_folder):int((current_folder + 1) * len(X) / total_folder)]
        if current_folder == 0:
            train_X = X[int(len(X) / total_folder):len(X)]
            train_Y = Y[int(len(X) / total_folder):len(X)]
        elif current_folder + 1 == self.k_folder:
            train_X = X[0:int(current_folder * len(X) / total_folder)]
            train_Y = Y[0:int(current_folder * len(X) / total_folder)]
        else:
            train_X = X[0:int(current_folder * len(X) / total_folder)]
            train_Y = Y[0:int(current_folder * len(X) / total_folder)]
            train_X_add = X[int((current_folder + 1) * len(X) / total_folder):len(X)]
            train_Y_add = Y[int((current_folder + 1) * len(X) / total_folder):len(X)]
            train_X = np.append(train_X, train_X_add)
            train_X = train_X.reshape(-1, 2)
            train_Y = np.append(train_Y, train_Y_add)
            train_Y = train_Y.reshape(-1, 1)
        return train_X, train_Y, test_X, test_Y

    def _forward_with_k_folder_ver_2(self):
        assert type(self.k_folder) == int
        assert self.k_folder_test == True
        assert self.k_folder <= int(len(self.data) / 2)
        for i in range(self.k_folder):
            # 此处执行数据集划分
            print('k-folder=', i + 1, '\ttotal=', self.k_folder)
            train_X, train_Y, test_X, test_Y = self._split_k_folder(i, self.k_folder, self.data, self.label)
            self.model = self._train(train_X, train_Y)
            if self.save_model:
                str_name = "model-folder(" + str(i + 1) + "with" + str(self.k_folder) + ".pkl"
                self._dump_model(True, str_name)
            self.y_pred = self._model_predict(test_X)
            y_pred = np.array(self.y_pred)
            target_name = ['0', '1']
            print(classification_report(test_Y, y_pred, target_names=target_name))
            str_name = "P-R(curve)-folder(" + str(i + 1) + "with" + str(self.k_folder)
            self._draw_P_R(test_Y, y_pred, change_name=True, new_name=str_name)
            print('混淆矩阵为：')
            str_name = "Confusion Matrix(" + str(i + 1) + "with" + str(self.k_folder)
            self._draw_confusion_matrix(test_Y, y_pred, change_name=True, new_name=str_name)

    def _make_dataset_ver_3(self):
        if not self.data_enhancement:
            iris = load_iris()
            X = iris.data
            Y = iris.target
            X = X[Y < 3, :2]
            Y = Y[Y < 3]
            # 标准化
            std = StandardScaler()
            std.fit(X)
            X_Standard = std.transform(X)
            return X_Standard, Y
        else:
            iris = load_iris()
            X = iris.data
            Y = iris.target
            X = X[Y < 3, :2]
            Y = Y[Y < 3]
            # enhance here
            assert type(self.enhancement_level) == int
            assert len(X) == len(Y)
            length = len(X)
            for i in range(self.enhancement_level):
                for j in range(length):
                    X_0 = X[j]
                    X_00 = X_0[0]
                    X_01 = X_0[1]
                    ran_0 = 0.01 / (self.enhancement_level + 1)
                    X = np.append(X, (
                        X_00 + random.uniform(-self.enhancement_level * ran_0, self.enhancement_level * ran_0),
                        X_01 + random.uniform(-self.enhancement_level * ran_0, self.enhancement_level * ran_0)))
                    Y = np.append(Y, Y[j])
                    X = X.reshape(-1, 2)
            # 标准化
            std = StandardScaler()
            std.fit(X)
            X_standard = std.transform(X)
            return X_standard, Y

    def _forward_ver_3(self):
        self.data, self.label = self._make_dataset_ver_3()
        if self.random_set:
            self.data, self.label = self._random_setting()
        if self.k_folder_test:
            return self._forward_with_k_folder_ver_3()
        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = self._split_dataset()
        return self._forward_none_k_folder_ver_3()

    def _update_label_for_kmeans(self,train_x,train_y,test_x,test_y):
        """
        :param train_x:
        :param train_y:
        :return: y_pred
        """
        """
        @ 数据更新方式 -> 更新标签
        @ step 1 -> 计算预测中心
        @ step 2 -> 计算训练中心
        @ step 3 -> 计算距离，中心对应，标签转换
        @ step 4 -> 更新标签
        """
        y_pred = []
        pred_label_0 = -1           #与输出标签0对应的真实标签
        pred_label_1 = -1           #与输出标签1对应的真实标签
        pred_label_2 = -1           #与输出标签2对应的真实标签
        # 计算预测中心
        pred_y_train = self._model_predict(train_x)
        pred_y_train_1 = self._model_predict(train_x)
        for i in range(len(pred_y_train)):
            if pred_y_train_1[i] != pred_y_train[i]:
                raise ValueError('outputError')
        pred_length_0 = 0
        pred_length_1 = 0
        pred_length_2 = 0
        temp_0_x,temp_0_y,temp_1_x,temp_1_y,temp_2_x,temp_2_y = 0,0,0,0,0,0
        for i in range(len(pred_y_train)):
            if pred_y_train[i] == 0:
                pred_length_0 += 1
                temp_0_x += train_x[i][0]
                temp_0_y += train_x[i][1]
            elif pred_y_train[i] == 1:
                pred_length_1 += 1
                temp_1_x += train_x[i][0]
                temp_1_y += train_x[i][1]
            elif pred_y_train[i] == 2:
                pred_length_2 += 1
                temp_2_x += train_x[i][0]
                temp_2_y += train_x[i][1]
            else:
                return
        pred_0_x = temp_0_x/pred_length_0
        pred_0_y = temp_0_y/pred_length_0
        pred_1_x = temp_1_x/pred_length_1
        pred_1_y = temp_1_y/pred_length_1
        pred_2_x = temp_2_x/pred_length_2
        pred_2_y = temp_2_y/pred_length_2
        # 计算真实标签中心
        length_0 = 0
        length_1 = 0
        length_2 = 0
        temp_00_x,temp_00_y,temp_11_x,temp_11_y,temp_22_x,temp_22_y = 0,0,0,0,0,0
        for i in range(len(train_x)):
            if train_y[i] == 0:
                length_0 += 1
                temp_00_x += train_x[i][0]
                temp_00_y += train_x[i][1]
            elif train_y[i] == 1:
                length_1 += 1
                temp_11_x += train_x[i][0]
                temp_11_y += train_x[i][1]
            elif train_y[i] == 2:
                length_2 += 1
                temp_22_x += train_x[i][0]
                temp_22_y += train_x[i][1]
            else:
                return
        true_0_x = temp_00_x/length_0
        true_0_y = temp_00_y/length_0
        true_1_x = temp_11_x/length_1
        true_1_y = temp_11_y/length_1
        true_2_x = temp_22_x/length_2
        true_2_y = temp_22_y/length_2
        # 计算距离
        label_0to0 = self._calc_distance(true_0_x,true_0_y,pred_0_x,pred_0_y)
        label_1to0 = self._calc_distance(true_0_x,true_0_y,pred_1_x,pred_1_y)
        label_2to0 = self._calc_distance(true_0_x,true_0_y,pred_2_x,pred_2_y)
        label_0to1 = self._calc_distance(true_1_x,true_1_y,pred_0_x,pred_0_y)
        label_1to1 = self._calc_distance(true_1_x,true_1_y,pred_1_x,pred_1_y)
        label_2to1 = self._calc_distance(true_1_x,true_1_y,pred_2_x,pred_2_y)
        label_0to2 = self._calc_distance(true_2_x,true_2_y,pred_0_x,pred_0_y)
        label_1to2 = self._calc_distance(true_2_x,true_2_y,pred_1_x,pred_1_y)
        label_2to2 = self._calc_distance(true_2_x,true_2_y,pred_2_x,pred_2_y)
        print(label_0to0,label_1to0,label_2to0)
        print(label_0to1,label_1to1,label_2to1)
        print(label_0to2,label_1to2,label_2to2)
        # 计算标签对应关系
        # 计算预测标签0对应的真实标签
        temp = np.array([label_0to0,label_0to1,label_0to2])
        index = np.argmin(temp)
        pred_label_0 = index
        # 计算预测标签1对应的真实标签
        temp = np.array([label_1to0,label_1to1,label_1to2])
        index = np.argmin(temp)
        pred_label_1 = index
        temp = np.array([label_2to0,label_2to1,label_2to2])
        index = np.argmin(temp)
        pred_label_2 = index
        print('标签对应计算完成：0->',pred_label_0,'\t1->',pred_label_1,'\t2->',pred_label_2)
        assert pred_label_2 != pred_label_1
        assert pred_label_2 != pred_label_0
        assert pred_label_1 != pred_label_0
        test_y = self._model_predict(test_x)
        for i in range(len(test_y)):
            if test_y[i] == 0:
                y_pred.append(pred_label_0)
            elif test_y[i] == 1:
                y_pred.append(pred_label_1)
            elif test_y[i] == 2:
                y_pred.append(pred_label_2)
            else:
                raise ValueError('can not update label in this solution')
        y_pred = np.array(y_pred)
        return y_pred

    def _calc_distance(self,x1,y1,x2,y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def _forward_none_k_folder_ver_3(self):
        train_x, train_y, val_x, val_y, test_x, test_y = self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y
        self.model = self._train(train_x, train_y)
        if self.save_model:
            self._dump_model()
        self.y_pred = self._model_predict(test_x)
        y_pred = self.y_pred
        y_pred = np.array(y_pred)
        target_name = ['0', '1', '2']
        """
        @ v4.0 算法更新，根据聚类结果以及样本点中心，重新分配标签
        """
        y_pred = self._update_label_for_kmeans(train_x,train_y,test_x,test_y)
        print(classification_report(test_y, y_pred, target_names=target_name))
        self._draw_P_R(test_y, y_pred)
        print('混淆矩阵为：')
        self._draw_plt_confusion_matrix(test_y, y_pred, num_classes=3, classes=['0', '1', '2'], add_acc=True)
        print('绘制聚类可视化效果图，请稍后')
        self._draw_test_info(test_x, test_y, y_pred, title='KNN model', add_acc=True)
        print('绘制完成，正在计算K-Means评估指标')
        self._judge_k_means(train_x,train_y)

    def _forward_with_k_folder_ver_3(self):
        assert type(self.k_folder) == int
        assert self.k_folder_test == True
        assert self.k_folder <= int(len(self.data) / 3)
        for i in range(self.k_folder):
            # 此处执行数据集划分
            print('K-folder=', i + 1, '\ttotal=', self.k_folder)
            train_X, train_Y, test_X, test_Y = self._split_k_folder(i, self.k_folder, self.data, self.label)
            self.model = self._train(train_X, train_Y)
            if self.save_model:
                str_name = "model-folder(" + str(i + 1) + "with" + str(self.k_folder) + ".pkl"
                self._dump_model(True, str_name)
            self.y_pred = self._model_predict(test_X)
            y_pred = np.array(self.y_pred)
            target_name = ['0', '1', '2']
            """
            @ v4.0 算法更新，根据聚类结果以及样本点中心，重新分配标签
            """
            y_pred = self._update_label_for_kmeans(train_X, train_Y, test_X, test_Y)
            print(classification_report(test_Y, y_pred, target_names=target_name))
            str_name = "P-R(curve)-folder(" + str(i + 1) + "with" + str(self.k_folder)
            self._draw_P_R(test_Y, y_pred, change_name=True, new_name=str_name)
            print('混淆矩阵为：')
            str_name = "Confusion Matrix(" + str(i + 1) + "with" + str(self.k_folder)
            self._draw_plt_confusion_matrix(test_Y, y_pred, title=str_name, add_acc=True, num_classes=3,
                                            classes=['0', '1', '2'])
            print('绘制聚类可视化效果图，请稍后')
            str_name = "KNN model " + str(i + 1) + "with" + str(self.k_folder)
            self._draw_test_info(test_X, test_Y, y_pred, title=str_name, add_acc=True)
            print('绘制完成，正在计算K-Means评估指标')
            self._judge_k_means(train_X,train_Y)

    def _draw_test_info(self, test_x, test_y, y_pred, title='Zzz', add_acc=False):
        if title == 'Zzz':
            title = 'model test<default draw>'
        clusters_number = 3
        fig, ax = plt.subplots()
        types = []
        colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(clusters_number)])
        colors = [rgb2hex(x) for x in colors]
        for i, color in enumerate(colors):
            need_idx = np.where(y_pred == i)[0]
            for idx in need_idx:
                ax.scatter(test_x[idx][0], test_x[idx][1], c=color, label=i)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        if add_acc:
            acc = 0
            total = 0
            for i in range(len(y_pred)):
                if y_pred[i] == test_y[i]:
                    acc += 1
                total += 1
            plt.title(title + ",acc=" + str(acc / total))
            if self.broker:
                plt.show()
            plt.savefig(title + ",acc=" + str(acc / total) + ".png")
        else:
            plt.title(title)
            if self.broker:
                plt.show()
            plt.savefig(title + ".png")
        plt.close()

    def _judge_k_means(self,X,Y):
        """
        模型评估指标
        :return: None
        """
        km = KMeans(n_clusters=3).fit(X,Y)
        sc = silhouette_score(X,km.labels_)
        print('评估指标：\ninertia=',km.inertia_,'\tsilhoutte_score=',sc)

def main_new():
    choose(2,False)

def main_old():
    t1 = time.time()
    # help(process(5, True, 200, True, 0.8, 0.0, 0.2, True, 1,True,True,False,True,'model.pkl','linear',get_help=False)) # 5折交叉验证,实验2
    # help(process(1, False, 200, True, 0.8, 0.0, 0.2, True, 1,True,True,False,True,'model.pkl','linear',get_help=False)) # 普通训练，实验2
    help(process(5, True, 200, True, 0.8, 0.0, 0.2, True, 1, True, True, False, True, 'model.pkl', 'linear',
                 get_help=False, experiment_ver=3, n_clusters=1))  # 5折交叉验证，实验3
    # help(process(1, False, 200, True, 0.8, 0.0, 0.2, True, 1,True,True,False,True,'model.pkl','linear',get_help=False,experiment_ver=3,n_clusters=1)) # 普通训练，实验3
    # a.help()
    t2 = time.time()
    print('time used:(sec)', t2 - t1)

if __name__ == '__main__':
    debug = True
    if debug:
        import cProfile
        cProfile.run('main_new()','debug_timer_rec')
    else :
        main_new()
