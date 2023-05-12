# 本python脚本可以应用于
## 1.SVM
## 2.KMeans
## 算法

# 可以用于机器学习实验：
## 1.基于聚类算法的安德森鸢尾花卉分类
## 2.基于支持向量机的安德森鸢尾花卉分类

# 可选选项
## 1.切换实验内容（自适应切换）
## 2.选择交叉验证方案

# 函数功能
## process.__init__() function:
##            usage:
##                k_folder :
##                    if k_folder_test == False ,this option is unavailable,else use cross validation(k-folder),default = 1
##                k_folder_test :
##                    if this option == True , we will use cross validation(k_folder),else,we will use common training way,default = False
##                enhancement_level :
##                    if data_enhancement == False ,this option is unavailable, else we will add noise on each data,default = 0
##                data_enhancement :
##                    if this option == False, we will never add noice on data,however this option may cause accuracy reducing,else we will add some noise to the data
##                size_train :
##                    if k_folder_test == True, this option is unavailable,else we will split dataset -> train_data(length = int(len(dataset)*size_train)),default = 0.6
##                size_val :
##                    if k_folder_test == True, this option is unavailable, else we will split dataset -> val_data(length = int(len(dataset)*size_val)),default = 0.2
##                size_test :
##                    if k_folder_test == True, this option is unavailable, else we will split dataset -> test_data(length = int(len(dataset)*size_test)),default = 0.2
##                random_set :
##                    if random_set == True, we will disrupt dataset according to the random_seed, else we will never shuffle the dataset, default = True
##                random_seed :
##                    if random_set == False, this option is unavailable, else we will disrupt dataset according to the random_seed, default = 1
##                draw_fusion_matrix :
##                    if draw_fusion_matrix == True, we will draw confusion matrix,else we will never draw confusion matrix, default = False
##                save_P_R :
##                    if save_P_R == True, we will save P-R curve by matplotlib.pyplot ,else we will never save P_R curve to your computer,default = False
##                runtime_broker :
##                    if runtime_broker == True, we will stop when the photograph is OK,else we will save the photograph directly,default = False
##                save_model :
##                    if save_model == True, we will dump model to your computer (current path) by joblib,else we will not dump model,default = True
##                model_name :
##                    if save_model == False, this option is unavailable,else we will use this name to save model, default = 'model.pkl'
##                kernel :
##                    SupportVectorMachine Kernel function -> default = 'linear'
##                help :
##                    get_help,default = False
##                experiment_ver :
##                    experiment version you can choose 2 or 3 to train with different type of model (exp_2 : SupportVectorMechine,exp_3 : K-NearestNeighbor)
##                n_neighbor :
##                    define n_neighbor(from sklearn.neighbor.KNeighborsClassifier(n_neighbor)) only experiment_ver == 3,this choice is available

# copyright
## this file is edited by lry 
## all rights reserved (c) 2020~2023
