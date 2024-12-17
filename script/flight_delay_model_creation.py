# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:32 2023

@author: ythiriet
"""

# Clearing all variables
try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
except:
    pass


# Global importation
import math
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import optuna
import random
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import shap
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTEN
import joblib


# Class containing all parameters
class Parameters():
    def __init__(self):
        self.CLEAR_MODE = True

        self.SWITCH_DATA_NOT_ENOUGHT = True
        self.NB_DATA_NOT_ENOUGHT = 1500

        self.RF_MODEL = False
        self.RF_MODEL_OPTI = True
        self.RF_MODEL_TRIAL = 80

        self.NN_MODEL = False
        self.NN_MODEL_OPTI = False
        self.NN_MODEL_TRIAL = 20

        self.GB_MODEL = True
        self.GB_MODEL_OPTI = False
        self.GB_MODEL_TRIAL = 50

        self.XG_MODEL = False
        self.XG_MODEL_OPTI = False
        self.XG_MODEL_TRIAL = 10

        self.SWITCH_PLOT_DATA = True
        self.SWITCH_REMOVING_DATA = True
        self.List_data_drop = ["DepTime"]
        self.SWITCH_DATA_REDUCTION = True

        self.SWITCH_ABERRANT_IDENTICAL_DATA = True
        self.SWITCH_RELATION_DATA = False
        self.Array_Relation_Data = np.array(
            [["Height", 2],["Age", 2]],
            dtype = object)

        self.SWITCH_ENCODE_DATA_PREDICT = True
        self.SWITCH_ENCODE_DATA = True
        self.SWITCH_ENCODE_DATA_ONEHOT = False
        self.List_Data_Encode = ["Geography", "Gender"]
        self.Array_Data_Encode_Replacement = np.zeros(4, dtype = object)
        self.Array_Data_Encode_Replacement[0] = np.array(
            [["Male",1],["Female",0]], dtype = object)
        self.Array_Data_Encode_Replacement[1] = np.array(
            [["yes",1],["no",0]], dtype = object)
        self.Array_Data_Encode_Replacement[2] = np.array(
            [["Always",3],["Frequently",2],["Sometimes",1],["no",0]], dtype = object),

        self.SWITCH_EQUILIBRATE_DATA = True
        self.SWITCH_SMOTEN_DATA = False

        self.SWITCH_SAMPLE_DATA = False
        self.Fraction_Sample_Data = 0.25

        self.NAME_DATA_PREDICT = "dep_delayed_15min"
        self.GENERIC_NAME_DATA_PREDICT = "plane delay" # for plot

        self.MULTI_CLASSIFICATION = False

        self.N_SPLIT = 5
        self.k_folds = KFold(n_splits=self.N_SPLIT)


    # Determining if multi-classification
    def Multi_Classification_Analysis(self, UNIQUE_PREDICT_VALUE):

        if UNIQUE_PREDICT_VALUE.shape[0] > 2:
            self.MULTI_CLASSIFICATION = True


class Data_Preparation():
    def __init__(self):
        self.Train_DataFrame = []
        self.Test_DataFrame = []
        self.Train_Stats = []
        self.Unique_Predict_Value = []
        self.Train_Correlation = []
        self.Duplicate_Line = []

        self.ARRAY_REPLACEMENT_ALL = np.zeros([0], dtype = object)
        self.INDEX_REPLACEMENT_ALL = np.zeros([0], dtype = object)


    def data_import(self, NAME_DATA_PREDICT):

        self.Train_DataFrame = pd.read_csv("./data/train.csv")
        self.Test_DataFrame = pd.read_csv("./data/test.csv")
        self.Train_Stats = self.Train_DataFrame.describe()


    def real_value_replacement(self):
        TABLE_REPLACEMENT = np.array([["9E","Easyjet"], ["AA","AirFrance"], ["AQ","Ryanair"], ["AS","Vueling"],
                             ["B6","Transavia"], ["CO","Lufthansa"], ["DH","Iberia"], ["DL","Wizzair"],
                             ["EV","Etihad"], ["F9","Emirates"], ["FL","Luxair"], ["HA","Peach"],
                             ["HP","StarFlyer"], ["MQ","MyAir"], ["NW","CityJet"], ["OH","AirIndia"],
                             ["OO","Afrijet"], ["TZ","HOP"], ["UA","UnitedAirlines"], ["US","ContinentalAirlines"],
                             ["WN","AmericanAirlines"], ["XE","BritishAirways"], ["YV","Aeroflot"]])

        for i in range(TABLE_REPLACEMENT.shape[0]):
            self.Train_DataFrame["UniqueCarrier"] = self.Train_DataFrame["UniqueCarrier"].replace(
                TABLE_REPLACEMENT[i,0], TABLE_REPLACEMENT[i,1])
            self.Test_DataFrame["UniqueCarrier"] = self.Test_DataFrame["UniqueCarrier"].replace(
                TABLE_REPLACEMENT[i,0], TABLE_REPLACEMENT[i,1])


    def data_predict_description(self, NAME_DATA_PREDICT):
        self.Unique_Predict_Value = self.Train_DataFrame.groupby(NAME_DATA_PREDICT)[NAME_DATA_PREDICT].count()
        
        # Printing first values
        print(self.Train_DataFrame.head())


    def data_encoding_replacement(self, Array_Replacement):

        for i_encoding, DataFrame in enumerate([self.Train_DataFrame, self.Test_DataFrame]):

            # Replacement
            for j in range(Array_Replacement.shape[0]):
                DataFrame = DataFrame.replace(Array_Replacement[j,0], Array_Replacement[j,1])

            # Recording the replacement
            if i_encoding == 0:
                self.Train_DataFrame = DataFrame
            else:
                self.Test_DataFrame = DataFrame


    def data_encoding_replacement_important(self, COLUMN_NAME):

        # Init
        self.ARRAY_REPLACEMENT_ALL = np.append(self.ARRAY_REPLACEMENT_ALL,
                                               np.zeros([1], dtype = object),
                                               axis = 0)
        self.INDEX_REPLACEMENT_ALL = np.append(self.INDEX_REPLACEMENT_ALL,
                                               np.zeros([1], dtype = object),
                                               axis = 0)

        DF_TRAIN_TEST = pd.concat([Global_Data.Train_DataFrame, Global_Data.Test_DataFrame],
                                  ignore_index = True)
        UNIQUE_DF_TRAIN_TEST = DF_TRAIN_TEST.groupby(COLUMN_NAME)[COLUMN_NAME].count()
        ARRAY_REPLACEMENT = pd.DataFrame(UNIQUE_DF_TRAIN_TEST.index).to_numpy()
        INDEX_REPLACEMENT = pd.DataFrame(UNIQUE_DF_TRAIN_TEST.index).index.tolist()

        for i_encoding, DataFrame in enumerate([self.Train_DataFrame, self.Test_DataFrame]):

            # Replacement
            for j in range(ARRAY_REPLACEMENT.shape[0]):
                DataFrame[COLUMN_NAME] = DataFrame[COLUMN_NAME].replace(ARRAY_REPLACEMENT[j],
                                                                        INDEX_REPLACEMENT[j])

            # Recording the replacement
            if i_encoding == 0:
                self.Train_DataFrame[COLUMN_NAME] = DataFrame[COLUMN_NAME]
            else:
                self.Test_DataFrame[COLUMN_NAME] = DataFrame[COLUMN_NAME]

        # Recording the replacement
        self.ARRAY_REPLACEMENT_ALL[-1] = ARRAY_REPLACEMENT
        self.INDEX_REPLACEMENT_ALL[-1] = INDEX_REPLACEMENT


    def data_encoding_replacement_predict(self, Array_Replacement):
        for j in range(Array_Replacement.shape[0]):
            self.Train_DataFrame = self.Train_DataFrame.replace(
                Array_Replacement[j,0],Array_Replacement[j,1])


    def data_encoding_onehot(self, Name_Data_Encode):
        Enc = OneHotEncoder(handle_unknown='ignore')
        Data_Encode_Train = self.Train_DataFrame.loc[:,[Name_Data_Encode]]
        Data_Encode_Test = self.Test_DataFrame.loc[:,[Name_Data_Encode]]
        Data_Encode_Name = Name_Data_Encode + Data_Encode_Train.groupby(Name_Data_Encode)[Name_Data_Encode].count().index
        Enc.fit(Data_Encode_Train)

        Data_Encode_Train = Enc.transform(Data_Encode_Train).toarray()
        Data_Encode_Train = pd.DataFrame(Data_Encode_Train,
                                         columns = Data_Encode_Name)

        self.Train_DataFrame = self.Train_DataFrame.drop(columns = Name_Data_Encode)
        self.Train_DataFrame = pd.concat([self.Train_DataFrame, Data_Encode_Train],
                                         axis = 1)

        Data_Encode_Test = Enc.transform(Data_Encode_Test).toarray()
        Data_Encode_Test = pd.DataFrame(Data_Encode_Test,
                                         columns = Data_Encode_Name)
        Data_Encode_Test = Data_Encode_Test.set_index(self.Test_DataFrame.index)

        self.Test_DataFrame = self.Test_DataFrame.drop(columns = Name_Data_Encode)
        self.Test_DataFrame = pd.concat([self.Test_DataFrame, Data_Encode_Test],
                                         axis = 1)


    def data_drop(self, Name_data_drop):
        self.Train_DataFrame = self.Train_DataFrame.drop([Name_data_drop],axis=1)
        self.Test_DataFrame = self.Test_DataFrame.drop([Name_data_drop],axis=1)


    def data_pow(self, Name_Data_Duplicate, Number_Duplicate):
        self.Train_DataFrame[Name_Data_Duplicate] = (
            self.Train_DataFrame[Name_Data_Duplicate].pow(Number_Duplicate))
        self.Test_DataFrame[Name_Data_Duplicate] = (
            self.Test_DataFrame[Name_Data_Duplicate].pow(Number_Duplicate))


    def data_duplicate_removal(self, NAME_DATA_PREDICT, Column_Drop = ""):

        if len(Column_Drop) == 0:
            Duplicated_Data_All = self.Train_DataFrame.drop(
                NAME_DATA_PREDICT, axis = 1).duplicated()
        else:
            Duplicated_Data_All = self.Train_DataFrame.drop(
                [Column_Drop, NAME_DATA_PREDICT],axis = 1).duplicated()
        self.Duplicate_Line = Duplicated_Data_All.loc[Duplicated_Data_All == True]
        self.Train_DataFrame = self.Train_DataFrame.drop(self.Duplicate_Line.index)
        
        # Information to the user
        print(f"{self.Duplicate_Line.shape[0]} has been removed because of duplicates")
        plot.pause(3)


    def remove_low_data(self, NB_DATA_NOT_ENOUGHT, NAME_DATA_NOT_ENOUGHT, LIST_NAME_DATA_REMOVE_MULTIPLE = []):

        # Searching for data with low values
        TRAIN_GROUP_VALUE = self.Train_DataFrame.groupby(NAME_DATA_NOT_ENOUGHT)[NAME_DATA_NOT_ENOUGHT].count().sort_values(ascending = False)

        # Adding values only inside NAME DATA REMOVE MULTIPLE
        for NAME_DATA_REMOVE_MULTIPLE in LIST_NAME_DATA_REMOVE_MULTIPLE:
            TRAIN_GROUP_VALUE_OTHER = self.Train_DataFrame.groupby(NAME_DATA_REMOVE_MULTIPLE)[NAME_DATA_REMOVE_MULTIPLE].count().index

        for VALUE_OTHER in TRAIN_GROUP_VALUE_OTHER:
            if np.sum(VALUE_OTHER == np.array(TRAIN_GROUP_VALUE.index)) == 0:
                TRAIN_GROUP_VALUE = pd.concat([TRAIN_GROUP_VALUE, pd.Series(0, index = [VALUE_OTHER])])

        # Searching for values to drop following number of elements
        REMOVE_TRAIN_GROUP_VALUE = TRAIN_GROUP_VALUE.drop(TRAIN_GROUP_VALUE[TRAIN_GROUP_VALUE > NB_DATA_NOT_ENOUGHT].index)

        # Removing data inside train and test dataframe
        for DATA_REMOVE in REMOVE_TRAIN_GROUP_VALUE.index:
            self.Train_DataFrame = self.Train_DataFrame.drop(self.Train_DataFrame[self.Train_DataFrame[NAME_DATA_NOT_ENOUGHT] == DATA_REMOVE].index)
            self.Test_DataFrame = self.Test_DataFrame.drop(self.Test_DataFrame[self.Test_DataFrame[NAME_DATA_NOT_ENOUGHT] == DATA_REMOVE].index)

            for NAME_DATA_REMOVE_MULTIPLE in LIST_NAME_DATA_REMOVE_MULTIPLE:
                self.Train_DataFrame = self.Train_DataFrame.drop(self.Train_DataFrame[self.Train_DataFrame[NAME_DATA_REMOVE_MULTIPLE] == DATA_REMOVE].index)
                self.Test_DataFrame = self.Test_DataFrame.drop(self.Test_DataFrame[self.Test_DataFrame[NAME_DATA_REMOVE_MULTIPLE] == DATA_REMOVE].index)

        # Reseting index
        self.Train_DataFrame = self.Train_DataFrame.reset_index(drop = True)
        self.Test_DataFrame = self.Test_DataFrame.reset_index(drop = True)

    def oversampling(self, NAME_DATA_PREDICT, NB_DATA_NOT_ENOUGHT, Name_Data_Oversample = ""):

        self.Unique_Predict_Value = self.Train_DataFrame.groupby(NAME_DATA_PREDICT)[NAME_DATA_PREDICT].count()
        Max_Nb_Data = np.amax(self.Unique_Predict_Value.to_numpy())

        if len(Name_Data_Oversample) > 1:
            Global_Table_Train_Equilibrate = self.Unique_Predict_Value.loc[(
                self.Unique_Predict_Value.index == "Overweight_Level_II")]
        else:
            # Global_Table_Train_Equilibrate = self.Unique_Predict_Value.loc[
            #     self.Unique_Predict_Value > NB_DATA_NOT_ENOUGHT]
            Global_Table_Train_Equilibrate = self.Unique_Predict_Value.loc[self.Unique_Predict_Value < Max_Nb_Data]

        for i in range(Global_Table_Train_Equilibrate.shape[0]):
            Matrix_To_Add = np.zeros(
                [0, self.Train_DataFrame.shape[1]],
                dtype=object)
            DF_Reference = self.Train_DataFrame.loc[self.Train_DataFrame[NAME_DATA_PREDICT] == pd.DataFrame(
                Global_Table_Train_Equilibrate.index).iloc[i][0]]
            for j in range(Max_Nb_Data - Global_Table_Train_Equilibrate.iloc[i]):
                Matrix_To_Add = np.append(
                    Matrix_To_Add,
                    np.zeros([1, self.Train_DataFrame.shape[1]],
                              dtype=object),
                    axis=0)

                Matrix_To_Add[-1, :] = DF_Reference.iloc[
                    random.randint(0, DF_Reference.shape[0] - 1), :].to_numpy()

            DataFrame_To_Add = pd.DataFrame(
                Matrix_To_Add,
                columns=self.Train_DataFrame.columns)

            self.Train_DataFrame = pd.concat(
                [self.Train_DataFrame, DataFrame_To_Add],
                ignore_index=True)


    def data_sample(self, Sample_Fraction):

        self.Train_DataFrame = self.Train_DataFrame.sample(
            frac = Sample_Fraction, replace = False, random_state = 0)



class Data_Plot():
    def __init__(self):
        self.Box_Plot_Data_Predict = ""
        self.Box_Plot_Data_Available = ""
        self.Correlation_Plot = ""
        self.Train_DataFrame = []
        self.Train_Correlation = []
        self.Unique_Predict_Value = []


    def Box_Plot_Data_Predict_Plot(self,
                                   GENERIC_NAME_DATA_PREDICT):

        # Init
        fig, self.Box_Plot_Data_Predict = plot.subplots(2)
        plot.suptitle(f"Data count following {GENERIC_NAME_DATA_PREDICT}",
                      fontsize = 25,
                      color = "gold",
                      fontweight = "bold")

        # Horizontal bars for each possibilities
        self.Box_Plot_Data_Predict[0].barh(
            y = self.Unique_Predict_Value.index,
            width=self.Unique_Predict_Value,
            height=0.03,
            label=self.Unique_Predict_Value.index)

        # Cumulative horizontal bars
        Cumulative_Value = 0
        for i in range(self.Unique_Predict_Value.shape[0]):
            self.Box_Plot_Data_Predict[1].barh(
                y=1,
                width=self.Unique_Predict_Value.iloc[i],
                left = Cumulative_Value)
            self.Box_Plot_Data_Predict[1].text(
                x = Cumulative_Value + 100,
                y = 0.25,
                s = self.Unique_Predict_Value.index[i])
            Cumulative_Value += self.Unique_Predict_Value.iloc[i]
        self.Box_Plot_Data_Predict[1].set_ylim(0, 2)
        self.Box_Plot_Data_Predict[1].legend(
            self.Unique_Predict_Value.index.to_numpy(),
            ncol=int(self.Unique_Predict_Value.shape[0]/2),
            fontsize=6)


    def Box_Plot_Data_Available_Plot(self):

        # Init
        Nb_Line = math.ceil((self.Train_DataFrame.shape[1] - 1)/3)
        Nb_Column = math.ceil((self.Train_DataFrame.shape[1] - 1)/3)

        # Box Plot for all data
        fig, self.Box_Plot_Data_Available = plot.subplots(Nb_Line, Nb_Column)
        plot.suptitle("Boxplot for all data into the TRAIN dataset",
                      fontsize = 25,
                      color = "chartreuse",
                      fontweight = "bold")

        for i in range(Nb_Line):
            for j in range(Nb_Column):
                if i*Nb_Column + j < self.Train_DataFrame.shape[1]:
                    try:
                        self.Box_Plot_Data_Available[i, j].boxplot(
                            self.Train_DataFrame.iloc[:, [i*Nb_Column + j]])
                        self.Box_Plot_Data_Available[i, j].set_title(
                            self.Train_DataFrame.iloc[:, i*Nb_Column + j].name,
                            fontweight = "bold",
                            fontsize = 15)
                    except:
                        continue


    def Plot_Data_Relation(self,
                           Name_Data_x,
                           Name_Data_y):

        plot.figure()
        plot.scatter(self.Train_DataFrame[Name_Data_x],
                     self.Train_DataFrame[Name_Data_y])
        plot.suptitle(
            f"Relation between {Name_Data_x} and {Name_Data_y} variables",
            fontsize = 25,
            color = "darkorchid",
            fontweight = "bold")


    def Correlation_Plot_Plot(self):

        fig2, self.Correlation_Plot = plot.subplots()
        im = self.Correlation_Plot.imshow(
            self.Train_Correlation,
            vmin=-1,
            vmax=1,
            cmap="bwr")
        self.Correlation_Plot.figure.colorbar(im, ax=self.Correlation_Plot)
        self.Correlation_Plot.set_xticks(np.linspace(
            0, self.Train_DataFrame.shape[1] - 1, self.Train_DataFrame.shape[1]))
        self.Correlation_Plot.set_xticklabels(np.array(self.Train_DataFrame.columns, dtype = str),
                                              rotation = 45)
        self.Correlation_Plot.set_yticks(np.linspace(
            0, self.Train_DataFrame.shape[1] - 1, self.Train_DataFrame.shape[1]))
        self.Correlation_Plot.set_yticklabels(np.array(self.Train_DataFrame.columns, dtype = str))



class Data_Modelling():
    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.Y_predict = []
        self.Y_predict_proba = []
        self.K_predict = []
        self.K_predict_proba = []

        self.MODEL = ""
        self.Model_Name = ""
        self.Y_predict = []
        self.Y_test = []
        self.Nb_Correct_Prediction = 0
        self.Percentage_Correct_Prediction = 0
        self.score = 0
        self.Best_Params = np.zeros([1], dtype = object)


    def Splitting_Data(self,
                       Train_DataFrame,
                       GENERIC_NAME_DATA_PREDICT,
                       MULTI_CLASSIFICATION):


        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            Train_DataFrame.drop([GENERIC_NAME_DATA_PREDICT], axis=1),
            Train_DataFrame.loc[:, [GENERIC_NAME_DATA_PREDICT]],
            test_size=0.2,
            random_state=0)


        # Turning Y_train and Y_test into boolean if needed
        if MULTI_CLASSIFICATION == False:
            self.Y_train = self.Y_train.astype(bool)
            self.Y_test = self.Y_test.astype(bool)


    def Smoten_Sampling(self):

        sampler = SMOTEN(random_state = 0)
        self.X_train, self.Y_train = sampler.fit_resample(self.X_train, self.Y_train)


    def Result_Plot(self):

        # Plotting results
        X_plot = np.linspace(1, self.Y_test.shape[0], self.Y_test.shape[0])

        fig, ax = plot.subplots(2)
        ax[0].scatter(X_plot, self.Y_predict, color="green")
        ax[0].scatter(X_plot, self.Y_test, color="orange")
        ax[0].legend([
            f"Prediction from {self.Model_Name} Model", "Real Results"])
        ax[1].scatter(X_plot, abs(
            self.Y_test.astype(int) - self.Y_predict.astype(int)))
        ax[1].legend(
            [f"Score for {self.Model_Name} Model : {round(self.Percentage_Correct_Prediction,2)}"])


    def Result_Report(self):

        print('\n------------------ Confusion Matrix -----------------\n')
        print(confusion_matrix(self.Y_test, self.Y_predict))

        print('\n-------------------- Key Metrics --------------------')
        print('\nAccuracy: {:.3f}'.format(accuracy_score(self.Y_test, self.Y_predict)))
        print('Balanced Accuracy: {:.3f}\n'.format(balanced_accuracy_score(self.Y_test, self.Y_predict)))

        print('Micro Precision: {:.3f}'.format(precision_score(self.Y_test, self.Y_predict, average='micro')))
        print('Micro Recall: {:.3f}'.format(recall_score(self.Y_test, self.Y_predict, average='micro')))
        print('Micro F1-score: {:.3f}\n'.format(f1_score(self.Y_test, self.Y_predict, average='micro')))

        print('Macro Precision: {:.3f}'.format(precision_score(self.Y_test, self.Y_predict, average='macro')))
        print('Macro Recall: {:.3f}'.format(recall_score(self.Y_test, self.Y_predict, average='macro')))
        print('Macro F1-score: {:.3f}\n'.format(f1_score(self.Y_test, self.Y_predict, average='macro')))

        print('Weighted Precision: {:.3f}'.format(precision_score(self.Y_test, self.Y_predict, average='weighted')))
        print('Weighted Recall: {:.3f}'.format(recall_score(self.Y_test, self.Y_predict, average='weighted')))
        print('Weighted F1-score: {:.3f}'.format(f1_score(self.Y_test, self.Y_predict, average='weighted')))

        print('\n--------------- Classification Report ---------------\n')
        print(classification_report(self.Y_test, self.Y_predict))

        print('\n--------------- Imbalanced Report ---------------\n')
        print(classification_report_imbalanced(self.Y_test, self.Y_predict))


    def Result_Report_Plot(self):

        plot.figure(figsize = (10,8))
        plot.ylim(1,40)
        plot.text(0.02,39,'------------------ Confusion Matrix -----------------')
        plot.text(0.02,29, confusion_matrix(self.Y_test, self.Y_predict))
        plot.text(0.4,28,'-------------------- Key Metrics --------------------')
        plot.text(0.4,26,'Accuracy: {:.3f}'.format(accuracy_score(self.Y_test, self.Y_predict)))
        plot.text(0.4,24,'Balanced Accuracy: {:.3f}\n'.format(balanced_accuracy_score(self.Y_test, self.Y_predict)))
        plot.text(0.4,22,'Micro Precision: {:.3f}'.format(precision_score(self.Y_test, self.Y_predict, average='micro')))
        plot.text(0.4,20,'Micro Recall: {:.3f}'.format(recall_score(self.Y_test, self.Y_predict, average='micro')))
        plot.text(0.4,18,'Micro F1-score: {:.3f}\n'.format(f1_score(self.Y_test, self.Y_predict, average='micro')))
        plot.text(0.4,16,'Macro Precision: {:.3f}'.format(precision_score(self.Y_test, self.Y_predict, average='macro')))
        plot.text(0.4,14,'Macro Recall: {:.3f}'.format(recall_score(self.Y_test, self.Y_predict, average='macro')))
        plot.text(0.4,12,'Macro F1-score: {:.3f}\n'.format(f1_score(self.Y_test, self.Y_predict, average='macro')))
        plot.text(0.4,10,'Weighted Precision: {:.3f}'.format(precision_score(self.Y_test, self.Y_predict, average='weighted')))
        plot.text(0.4,8,'Weighted Recall: {:.3f}'.format(recall_score(self.Y_test, self.Y_predict, average='weighted')))
        plot.text(0.4,6,'Weighted F1-score: {:.3f}'.format(f1_score(self.Y_test, self.Y_predict, average='weighted')))
        plot.text(0.02,15,'--------------- Classification Report ---------------')
        plot.text(0.02,1,classification_report(self.Y_test, self.Y_predict))
        plot.suptitle(f"Various Result Score for {self.Model_Name}")
        plot.text(0.4,39,'--------------- Imbalanced Report ---------------')
        plot.text(0.4,29,classification_report_imbalanced(self.Y_test, self.Y_predict))



class Data_Modelling_Random_Forest(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_Random_Forest, self).__init__()

        self.Nb_Tree = 146
        self.min_samples_leaf = 16
        self.min_samples_split = 7
        self.min_weight_fraction_leaf = 0.00007276912136637689
        self.max_depth = 33
        
        self.Nb_Tree = 140
        self.min_samples_leaf = 5
        self.min_samples_split = 22
        self.min_weight_fraction_leaf = 0.00021061239694571002
        self.max_depth = 32

        self.Start_Point = 0
        self.End_Point_1 = 20
        self.End_Point_2 = 100
        self.shap_explainer = 0


    def Random_Forest_Modellisation(self, k_folds):

        # Setting the model with parameters
        self.MODEL = RandomForestClassifier(
            n_estimators=self.Nb_Tree,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            verbose=2,
            random_state=0)

        # Cross validation
        self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds)
        self.MODEL.fit(self.X_train, self.Y_train)

        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        self.Y_predict_proba = self.MODEL.predict_proba(self.X_test)
        self.Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        self.Nb_Correct_Prediction = np.count_nonzero(
            self.Y_test.astype(int) - self.Y_predict.astype(int))
        self.Percentage_Correct_Prediction = (1 -
            self.Nb_Correct_Prediction / self.Y_test.shape[0])
        print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")


    def Feature_Importance_Plot(self):

        # Feature Importance
        RF_Feature_Importance = pd.DataFrame(
            {'Variable': self.X_train.columns,
             'Importance': self.MODEL.feature_importances_}).sort_values(
                 'Importance', ascending=False)

        fig, ax = plot.subplots()
        ax.barh(RF_Feature_Importance.Variable,
                RF_Feature_Importance.Importance)
        plot.grid()
        plot.suptitle("Feature Importance for Random Forest Model")


    def Permutation_Importance(
            self, Test_DataFrame):

         # Permutation Importance
        Permutation_Importance_Train = permutation_importance(
            self.MODEL, self.X_train, self.Y_train,
            n_repeats=10, random_state=0, n_jobs=2)
        Permutation_Importance_Test = permutation_importance(
            self.MODEL, self.X_test, self.Y_test,
            n_repeats=10, random_state=0, n_jobs=2)

        # Init
        fig, ax = plot.subplots(2)
        max_importances = 0

        # Loop for Train/Test Data
        for i, Permutation_Importance in enumerate(
                [Permutation_Importance_Train, Permutation_Importance_Test]):

            # Calculating permutaion importance
            sorted_importances_idx = Permutation_Importance.importances_mean.argsort()
            importances = pd.DataFrame(
                Permutation_Importance.importances[sorted_importances_idx].T,
                columns=Test_DataFrame.columns[sorted_importances_idx],)
            max_importances = max([max_importances,importances.max().max()])

            # Plotting results
            ax[i].boxplot(importances, vert=False)
            ax[i].set_title("Permutation Importances")
            ax[i].axvline(x=0, color="k", linestyle="--")
            ax[i].set_xlabel("Decrease in accuracy score")
            ax[i].set_xlim([-0.01,max_importances + 0.1])
            ax[i].set_yticks(np.linspace(1,importances.shape[1],importances.shape[1]))
            ax[i].set_yticklabels(importances.columns)
            ax[i].figure.tight_layout()


    def Shap_Value_Analysis_Single_Point(self):

        # Init
        Nb_Analysed = random.randint(0,self.X_test.shape[0])

        # Create object that can calculate shap values
        self.shap_explainer = shap.TreeExplainer(self.MODEL)

        # Calculate Shap values for one point
        self.MODEL.predict(np.array(self.X_test.iloc[Nb_Analysed,:]).reshape(-1,1).T)
        shap_values_1 = self.shap_explainer.shap_values(self.X_test.iloc[Nb_Analysed,:])

        # Plotting results
        shap.initjs()
        shap.force_plot(
            self.shap_explainer.expected_value[0],
            shap_values_1[1],
            self.X_test.iloc[Nb_Analysed,:].index,
            matplotlib=True)
        plot.suptitle(f"Prediction attendue : {self.Y_test[Nb_Analysed]}")


    def Shap_Value_Analysis_Multiple_Point(self):

        # Calculate Shap values for 10 points
        Wrong_pred = self.MODEL.predict(self.X_test) != np.array(self.Y_test)
        shap_values_10 = self.shap_explainer.shap_values(
            self.X_test.iloc[self.Start_Point:self.End_Point_1,:])

        # Plotting results
        plot.figure()
        shap.decision_plot(
            self.shap_explainer.expected_value[0],
            shap_values_10[0],
            self.X_test.iloc[self.Start_Point:self.End_Point_1,:],
            feature_names = np.array(self.X_test.columns),
            highlight=Wrong_pred[self.Start_Point:self.End_Point_1])


    def Shap_Value_Analysis_Multiple_Massive_Point(self):

        # Calculate Shap values for 100 points
        shap_value_100 = self.shap_explainer.shap_values(
            self.X_test.iloc[self.Start_Point:self.End_Point_2,:])

        # Plotting results
        shap.dependence_plot(
            2,
            shap_value_100[0],
            self.X_test.iloc[self.Start_Point:self.End_Point_2,:])



class Data_Modelling_Gradient_Boosting(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_Gradient_Boosting, self).__init__()

        # self.learning_rate = 0.24008228971438916
        # self.Nb_Tree = 157
        # self.min_samples_leaf = 11
        # self.min_samples_split = 10
        # self.min_weight_fraction_leaf = 0.04422002096962485
        # self.max_depth = 26
        # self.validation_fraction = 0.1   # Early Stopping
        # self.n_iter_no_change = 10   # Early Stopping
        # self.train_errors = []   # Early Stopping
        # self.test_errors = []   # Early Stopping
        
        self.learning_rate = 0.4968286095170373
        self.Nb_Tree = 192
        self.min_samples_leaf = 37
        self.min_samples_split = 35
        self.min_weight_fraction_leaf = 0.00018800850129667424
        self.max_depth = 24
        self.validation_fraction = 0.1   # Early Stopping
        self.n_iter_no_change = 10   # Early Stopping
        self.train_errors = []   # Early Stopping
        self.test_errors = []   # Early Stopping


    def Gradient_Boosting_Modellisation(self, N_SPLIT):

        # Setting the model with parameters
        self.MODEL = GradientBoostingClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.Nb_Tree,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            verbose=2,
            random_state=0,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change)

        # Init
        k_folds = KFold(n_splits=N_SPLIT)

        # Cross validation
        self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds)
        self.MODEL.fit(self.X_train, self.Y_train)

        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        self.Y_predict_proba = self.MODEL.predict_proba(self.X_test)
        self.Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        self.Nb_Correct_Prediction = np.count_nonzero(
            self.Y_test.astype(int) - self.Y_predict.astype(int))
        self.Percentage_Correct_Prediction = (1 -
            self.Nb_Correct_Prediction / self.Y_test.shape[0])
        print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")


    def Plot_Training_Validation_Error(self):

        for i, (train_pred, test_pred) in enumerate(
            zip(
                self.MODEL.staged_predict(self.X_train),
                self.MODEL.staged_predict(self.X_test),
            )
        ):

            if isinstance(self.Y_train.iloc[0][0], bool):
                self.train_errors.append(mean_squared_error(
                    self.Y_train, train_pred))
                self.test_errors.append(mean_squared_error(
                    self.Y_test, test_pred))
            else:
                self.train_errors.append(mean_squared_error(
                    self.Y_train.astype(int), train_pred.astype(int)))
                self.test_errors.append(mean_squared_error(
                    self.Y_test.astype(int), test_pred.astype(int)))


        fig, ax = plot.subplots(ncols=2, figsize=(12, 4))

        ax[0].plot(self.train_errors, label="Gradient Boosting with Early Stopping")
        ax[0].set_xlabel("Boosting Iterations")
        ax[0].set_ylabel("MSE (Training)")
        ax[0].set_yscale("log")
        ax[0].legend()
        ax[0].set_title("Training Error")

        ax[1].plot(self.test_errors, label="Gradient Boosting with Early Stopping")
        ax[1].set_xlabel("Boosting Iterations")
        ax[1].set_ylabel("MSE (Validation)")
        ax[1].set_yscale("log")
        ax[1].legend()
        ax[1].set_title("Validation Error")



class Data_Modelling_Neural_Network(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_Neural_Network, self).__init__()

        self.n_hidden = 3
        self.n_neurons = 68
        self.n_trials = 20
        self.History = 0

        self.monitor ="val_accuracy"
        self.min_delta = 0.002
        self.patience = 25


    def build_model_NN(self):

        # Neural Network model
        self.MODEL = tf.keras.models.Sequential()
        self.MODEL.add(tf.keras.layers.InputLayer(input_shape=self.X_train.shape[1]))
        for layers in range(self.n_hidden):
            self.MODEL.add(tf.keras.layers.Dense(self.n_neurons, activation = "relu"))
        self.MODEL.add(tf.keras.layers.Dense(Global_Data.Unique_Predict_Value.shape[0], activation = "softmax"))

        # Optimizer used
        OPTIMIZER = tf.keras.optimizers.Nadam()

        self.MODEL.compile(loss = "sparse_categorical_crossentropy",
                      optimizer = OPTIMIZER,
                      metrics = "accuracy")


    # Creating Model with best params
    def Build_Best_Model_NN(self):

        # Early stopping init
        callback = tf.keras.callbacks.EarlyStopping(
            monitor = self.monitor,
            min_delta = self.min_delta,
            patience = self.patience,
            verbose = 0,
            restore_best_weights=True)

        # NN model creation with best parameters
        tf.random.set_seed(0)
        self.MODEL = build_model_NN(
            self.n_hidden,
            self.n_neurons,
            input_shape = (self.X_train.shape[1]))

        self.History = self.MODEL.fit(
            np.asarray(self.X_train).astype("float32"),
            np.asarray(self.Y_train).astype("float32"),
            epochs = 500,
            validatioN_SPLIT = 0.01,
            initial_epoch = 0,
            callbacks=[callback])


    # Predicting Results
    def Predicting_Results(self):

        self.Y_test = np.squeeze(self.Y_test.to_numpy())
        Y_Predict_Proba = self.MODEL.predict(np.asarray(self.X_test).astype("float32"))
        self.Y_predict = np.zeros([Y_Predict_Proba.shape[0]], dtype = float)

        for i in range(Y_Predict_Proba.shape[0]):
            self.Y_predict[i] = np.where(Y_Predict_Proba[i,:] == np.amax(Y_Predict_Proba[i,:]))[0]

        # Calculating Percentage of correct prediction
        self.Nb_Correct_Prediction = np.count_nonzero(self.Y_test.astype(int) - self.Y_predict.astype(int))
        self.Percentage_Correct_Prediction = 1 - self.Nb_Correct_Prediction / self.Y_test.shape[0]
        print(f" Pourcentage de predictions correctes {self.Model_Name} : {self.Percentage_Correct_Prediction} %")


    # Plot learning history
    def Plot_Learning_History(self):

        # Plot learning evolution for Neural Network
        pd.DataFrame(self.History.history).plot(figsize = (8,5))
        plot.grid(True)
        plot.title("Learning Evolution for Neural Network")



class Data_Modelling_XGBoosting(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_XGBoosting, self).__init__()

        self.objective='multi:softmax'
        self.num_class=16
        self.learning_rate=0.1
        self.max_depth=5
        self.gamma=0
        self.reg_lambda=1
        self.early_stopping_rounds=25
        self.eval_metric=['merror','mlogloss']

        self.x_axis = []
        self.results_metric_plot = []


    def XGBoosting_Modellisation(self, k_folds):

        # Setting the model with parameters
        self.MODEL = xgb.XGBClassifier(
            objective=self.objective,
            num_class=self.num_class,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            seed=42)

        # Cross validation
        # self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds)
        self.MODEL.fit(self.X_train, self.Y_train,
                       verbose = 1,
                       eval_set = [(self.X_train, self.Y_train),
                                   (self.X_test, self.Y_test)])

        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        self.Y_predict_proba = self.MODEL.predict_proba(self.X_test)
        self.Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        self.Nb_Correct_Prediction = np.count_nonzero(
            self.Y_test.astype(int) - self.Y_predict.astype(int))
        self.Percentage_Correct_Prediction = (1 -
            self.Nb_Correct_Prediction / self.Y_test.shape[0])
        print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")

        # Preparing evaluation metric plots
        self.results_metric_plot = self.MODEL.evals_result()
        epochs = len(self.results_metric_plot['validation_0']['mlogloss'])
        self.x_axis = range(0, epochs)


    def Evaluation_Metric_Plot_Mlogloss(self):

        # xgboost 'mlogloss' plot
        fig, ax = plot.subplots(figsize=(9,5))
        ax.plot(self.x_axis, self.results_metric_plot['validation_0']['mlogloss'], label='Train')
        ax.plot(self.x_axis, self.results_metric_plot['validation_1']['mlogloss'], label='Test')
        ax.legend()
        plot.ylabel('mlogloss')
        plot.title('GridSearchCV XGBoost mlogloss')
        plot.show()


    def Evaluation_Metric_Plot_Merror(self):

        # xgboost 'merror' plot
        fig, ax = plot.subplots(figsize=(9,5))
        ax.plot(self.x_axis, self.results_metric_plot['validation_0']['merror'], label='Train')
        ax.plot(self.x_axis, self.results_metric_plot['validation_1']['merror'], label='Test')
        ax.legend()
        plot.ylabel('merror')
        plot.title('GridSearchCV XGBoost merror')
        plot.show()


    def Feature_Importance_Plot(self):

        fig, ax = plot.subplots(figsize=(9,5))
        plot_importance(self.MODEL, ax=ax)
        plot.show()



# Init for global parameters
Global_Parameters = Parameters()

if Global_Parameters.CLEAR_MODE:

    # Closing all figures
    plot.close("all")


Global_Data = Data_Preparation()
Global_Data.data_import(Global_Parameters.NAME_DATA_PREDICT)

# Cheat
Global_Data.real_value_replacement()

# Removing variable with too low data
if Global_Parameters.SWITCH_DATA_REDUCTION:
    Global_Data.remove_low_data(Global_Parameters.NB_DATA_NOT_ENOUGHT, "Origin",
                                LIST_NAME_DATA_REMOVE_MULTIPLE = ["Dest"])

Global_Data.data_predict_description(Global_Parameters.NAME_DATA_PREDICT)
Global_Parameters.Multi_Classification_Analysis(Global_Data.Unique_Predict_Value)
Global_Data.Train_Correlation = Global_Data.Train_DataFrame.iloc[
    :,:Global_Data.Train_DataFrame.shape[1] - 1].corr()

# Getting number for month, dayofmonth and dayofweek
Global_Data.Train_DataFrame = Global_Data.Train_DataFrame.replace(regex = "c-", value = "")
Global_Data.Train_DataFrame["Month"] = Global_Data.Train_DataFrame["Month"].astype("int")
Global_Data.Train_DataFrame["DayofMonth"] = Global_Data.Train_DataFrame["DayofMonth"].astype("int")
Global_Data.Train_DataFrame["DayOfWeek"] = Global_Data.Train_DataFrame["DayOfWeek"].astype("int")
Global_Data.Test_DataFrame = Global_Data.Test_DataFrame.replace(regex = "c-", value = "")
Global_Data.Test_DataFrame["Month"] = Global_Data.Train_DataFrame["Month"].astype("int")
Global_Data.Test_DataFrame["DayofMonth"] = Global_Data.Test_DataFrame["DayofMonth"].astype("int")
Global_Data.Test_DataFrame["DayOfWeek"] = Global_Data.Test_DataFrame["DayOfWeek"].astype("int")

# Encoding data for entry variables
if Global_Parameters.SWITCH_ENCODE_DATA:
    if Global_Parameters.SWITCH_ENCODE_DATA_ONEHOT:
        for Name_Data_Encode in Global_Parameters.List_Data_Encode:
            Global_Data.data_encoding_onehot(Name_Data_Encode)


    else:
        Global_Data.data_encoding_replacement_important("UniqueCarrier")
        Global_Data.data_encoding_replacement_important("Origin")
        Global_Data.data_encoding_replacement_important("Dest")
        # for Array in Global_Parameters.Array_Data_Encode_Replacement:
        #     Global_Data.data_encoding_replacement(Array)


# Searching for and removing aberrant/identical values
if Global_Parameters.SWITCH_ABERRANT_IDENTICAL_DATA:
    Global_Data.data_duplicate_removal(Global_Parameters.NAME_DATA_PREDICT)


# oversampling to equilibrate data
if (Global_Parameters.SWITCH_EQUILIBRATE_DATA and
    Global_Parameters.SWITCH_SMOTEN_DATA == False):
    Global_Data.oversampling(Global_Parameters.NAME_DATA_PREDICT, Global_Parameters.NB_DATA_NOT_ENOUGHT)


# Encoding data for predict variable
if Global_Parameters.SWITCH_ENCODE_DATA_PREDICT:
    Global_Data.data_encoding_replacement_predict(
        np.array([["Y",1],["N",0]], dtype = object))


# Searching for repartition on data to predict
if Global_Parameters.SWITCH_PLOT_DATA:

    Global_Data_Plot = Data_Plot()
    Global_Data_Plot.Train_DataFrame = Global_Data.Train_DataFrame
    Global_Data_Plot.Unique_Predict_Value = Global_Data.Unique_Predict_Value
    Global_Data_Plot.Box_Plot_Data_Predict_Plot(Global_Parameters.GENERIC_NAME_DATA_PREDICT)
    Global_Data_Plot.Box_Plot_Data_Available_Plot()
    plot.pause(1)
    # Global_Data_Plot.Plot_Data_Relation("Height", "Gender")
    plot.pause(1)
    Global_Data.Train_Correlation = Global_Data.Train_DataFrame.iloc[
        :,:Global_Data.Train_DataFrame.shape[1] - 1].corr()
    Global_Data_Plot.Train_Correlation = Global_Data.Train_Correlation
    Global_Data_Plot.Correlation_Plot_Plot()




# Removing Data
if Global_Parameters.SWITCH_REMOVING_DATA:
    for Name_data_drop in Global_Parameters.List_data_drop:
        Global_Data.data_drop(Name_data_drop)


# Modifying linear relation between data
if Global_Parameters.SWITCH_RELATION_DATA:
    for i in range(Global_Parameters.List_Relation_Data.shape[0]):
        Global_Data.data_pow(Global_Parameters.List_Relation_Data[i,0],
                             Global_Parameters.List_Relation_Data[i,1])


# Sample Data
if Global_Parameters.SWITCH_SAMPLE_DATA:
    Global_Data.data_sample(Global_Parameters.Fraction_Sample_Data)


# Generic Data Model
Data_Model = Data_Modelling()
Data_Model.Splitting_Data(Global_Data.Train_DataFrame,
                          Global_Parameters.NAME_DATA_PREDICT,
                          Global_Parameters.MULTI_CLASSIFICATION)
if (Global_Parameters.SWITCH_SMOTEN_DATA and Global_Parameters.SWITCH_EQUILIBRATE_DATA):
    Data_Model.Smoten_Sampling()


#
# Random Forest

if Global_Parameters.RF_MODEL:
    DATA_MODEL_RF = Data_Modelling_Random_Forest()
    DATA_MODEL_RF.X_train = Data_Model.X_train
    DATA_MODEL_RF.Y_train = Data_Model.Y_train
    DATA_MODEL_RF.X_test = Data_Model.X_test
    DATA_MODEL_RF.Y_test = Data_Model.Y_test
    DATA_MODEL_RF.Model_Name = "Random Forest"

    # Building a Random Forest Model with adjusted parameters
    def build_model_RF(
            Nb_Tree=1,
            min_samples_leaf=2,
            min_samples_split=10,
            max_depth=2,
            min_weight_fraction_leaf=0.5):

        MODEL_RF = RandomForestClassifier(
            n_estimators=Nb_Tree,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=0,
            max_depth=max_depth,
            min_weight_fraction_leaf=min_weight_fraction_leaf,)

        return MODEL_RF


    # Searching for Optimized Hyperparameters
    if Global_Parameters.RF_MODEL_OPTI:

        # Building function to minimize
        def objective_RF(trial):
            params = {'Nb_Tree': trial.suggest_int('Nb_Tree', 10, 250),
                      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
                      'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                      'max_depth': trial.suggest_int('max_depth', 1, 50),
                      'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)}

            MODEL_RF = build_model_RF(**params)
            scores = cross_val_score(
                MODEL_RF, DATA_MODEL_RF.X_train, DATA_MODEL_RF.Y_train, cv=Global_Parameters.k_folds)
            MODEL_RF.fit(DATA_MODEL_RF.X_train, DATA_MODEL_RF.Y_train)
            prediction_score = MODEL_RF.score(DATA_MODEL_RF.X_test, DATA_MODEL_RF.Y_test)

            return 0.1*np.amax(scores) + prediction_score

        # Search for best parameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_RF, n_trials=Global_Parameters.RF_MODEL_TRIAL,
                       catch=(ValueError,))
        Best_params_RF = np.zeros([1], dtype=object)
        Best_params_RF[0] = study.best_params
        DATA_MODEL_RF.Nb_Tree = int(Best_params_RF[0].get("Nb_Tree"))
        DATA_MODEL_RF.min_samples_leaf = int(Best_params_RF[0].get("min_samples_leaf"))
        DATA_MODEL_RF.min_samples_split = int(Best_params_RF[0].get("min_samples_split"))
        DATA_MODEL_RF.min_weight_fraction_leaf = float(Best_params_RF[0].get("min_weight_fraction_leaf"))
        DATA_MODEL_RF.max_depth = int(Best_params_RF[0].get("max_depth"))

    DATA_MODEL_RF.Random_Forest_Modellisation(k_folds = Global_Parameters.k_folds)
    DATA_MODEL_RF.Result_Plot()
    DATA_MODEL_RF.Feature_Importance_Plot()
    DATA_MODEL_RF.Permutation_Importance(Global_Data.Test_DataFrame)
    DATA_MODEL_RF.Result_Report()
    DATA_MODEL_RF.Result_Report_Plot()

    DATA_MODEL_RF.Shap_Value_Analysis_Single_Point()
    DATA_MODEL_RF.Shap_Value_Analysis_Multiple_Point()
    DATA_MODEL_RF.Shap_Value_Analysis_Multiple_Massive_Point()


#
# Gradient Boosting

if Global_Parameters.GB_MODEL:
    DATA_MODEL_GB = Data_Modelling_Gradient_Boosting()
    DATA_MODEL_GB.X_train = Data_Model.X_train
    DATA_MODEL_GB.Y_train = Data_Model.Y_train
    DATA_MODEL_GB.X_test = Data_Model.X_test
    DATA_MODEL_GB.Y_test = Data_Model.Y_test
    DATA_MODEL_GB.Model_Name = "Gradient Boosting"

    # Building a Gradient Boosting Model with adjusted parameters
    def build_model_GB(
            learning_rate=0.1,
            Nb_Tree=1,
            min_samples_split=10,
            min_samples_leaf=2,
            min_weight_fraction_leaf=0.5,
            max_depth=2):

        MODEL_GB = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=Nb_Tree,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=0,
            max_depth=max_depth,
            min_weight_fraction_leaf=min_weight_fraction_leaf)

        return MODEL_GB

    # Searching for Optimized Hyperparameters
    if Global_Parameters.GB_MODEL_OPTI:

        # Building function to minimize
        def objective_GB(trial):
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
                      'Nb_Tree': trial.suggest_int('Nb_Tree', 2, 200),
                      'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),
                      'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
                      'max_depth': trial.suggest_int('max_depth', 2, 50)}

            MODEL_GB = build_model_GB(**params)
            scores = cross_val_score(MODEL_GB, Data_Model.X_train, Data_Model.Y_train, cv=Global_Parameters.k_folds)
            MODEL_GB.fit(Data_Model.X_train, Data_Model.Y_train)
            prediction_score = MODEL_GB.score(Data_Model.X_test, Data_Model.Y_test)

            return 0.1*np.amax(scores) + prediction_score

        # Search for best parameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_GB, n_trials=Global_Parameters.GB_MODEL_TRIAL,
                       catch=(ValueError,))
        Best_params_GB = np.zeros([1], dtype=object)
        Best_params_GB[0] = study.best_params
        DATA_MODEL_GB.learning_rate = float(Best_params_GB[0].get("learning_rate"))
        DATA_MODEL_GB.Nb_Tree = int(Best_params_GB[0].get("Nb_Tree"))
        DATA_MODEL_GB.min_samples_leaf = int(Best_params_GB[0].get("min_samples_leaf"))
        DATA_MODEL_GB.min_samples_split = int(Best_params_GB[0].get("min_samples_split"))
        DATA_MODEL_GB.min_weight_fraction_leaf = float(Best_params_GB[0].get("min_weight_fraction_leaf"))
        DATA_MODEL_GB.max_depth = int(Best_params_GB[0].get("max_depth"))

    DATA_MODEL_GB.Gradient_Boosting_Modellisation(
        N_SPLIT = Global_Parameters.N_SPLIT)
    DATA_MODEL_GB.Result_Plot()
    DATA_MODEL_GB.Plot_Training_Validation_Error()
    DATA_MODEL_GB.Result_Report_Plot()
    DATA_MODEL_GB.Result_Report()


#
# Neural Network

if Global_Parameters.NN_MODEL:
    DATA_MODEL_NN = Data_Modelling_Neural_Network()
    DATA_MODEL_NN.X_train = Data_Model.X_train
    DATA_MODEL_NN.Y_train = Data_Model.Y_train
    DATA_MODEL_NN.X_test = Data_Model.X_test
    DATA_MODEL_NN.Y_test = Data_Model.Y_test
    DATA_MODEL_NN.Model_Name = "Neural Network"

    def build_model_NN(
            n_hidden = 1,
            n_neurons = 100,
            input_shape = (Data_Model.X_train.shape[1])):

        # Neural Network model
        MODEL = tf.keras.models.Sequential()
        MODEL.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for layers in range(n_hidden):
            MODEL.add(tf.keras.layers.Dense(n_neurons, activation = "relu"))
        MODEL.add(tf.keras.layers.Dense(Global_Data.Unique_Predict_Value.shape[0], activation = "softmax"))

        # Optimizer used
        OPTIMIZER = tf.keras.optimizers.Nadam()

        MODEL.compile(loss = "sparse_categorical_crossentropy",
                      optimizer = OPTIMIZER,
                      metrics = "accuracy")

        return MODEL


    if Global_Parameters.NN_MODEL_OPTI:

        # Building function to minimize
        def objective_NN(trial):
            params = {
                'n_hidden': trial.suggest_int('n_hidden', 2, 4),
                'n_neurons': trial.suggest_int('n_neurons', 10, 100)}

            model = build_model_NN(**params)

            model.fit(
                np.asarray(DATA_MODEL_NN.X_train).astype("float32"),
                np.asarray(DATA_MODEL_NN.Y_train).astype("float32"),
                epochs = 20,
                validatioN_SPLIT = 0.01,
                initial_epoch = 0)

            Preds_NN_proba = model.predict(np.asarray(DATA_MODEL_NN.X_test).astype("float32"))
            Preds_NN = np.zeros([Preds_NN_proba.shape[0]], dtype = int)

            # Turning probability prediction into prediction
            for i in range(Preds_NN_proba.shape[0]):
                Preds_NN[i] = np.where(Preds_NN_proba[i,:] == np.amax(Preds_NN_proba[i,:]))[0][0]

            Score_NN = accuracy_score(Data_Model.Y_test, Preds_NN)

            return Score_NN


        # Search for best parameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_NN,
                       n_trials=Global_Parameters.NN_MODEL_TRIAL,
                       catch=(ValueError,))
        Best_params_NN = np.zeros([1], dtype = object)
        Best_params_NN[0] = study.best_params
        DATA_MODEL_NN.n_hidden = int(Best_params_NN[0].get("n_hidden"))
        DATA_MODEL_NN.n_neurons = int(Best_params_NN[0].get("n_neurons"))
    DATA_MODEL_NN.Build_Best_Model_NN()
    DATA_MODEL_NN.Plot_Learning_History()
    DATA_MODEL_NN.Predicting_Results()
    DATA_MODEL_NN.Result_Plot()
    DATA_MODEL_NN.Result_Report_Plot()
    DATA_MODEL_NN.Result_Report()


#
# XGBoosting

if Global_Parameters.XG_MODEL:
    DATA_MODEL_XG = Data_Modelling_XGBoosting()
    DATA_MODEL_XG.X_train = Data_Model.X_train
    DATA_MODEL_XG.Y_train = Data_Model.Y_train
    DATA_MODEL_XG.X_test = Data_Model.X_test
    DATA_MODEL_XG.Y_test = Data_Model.Y_test
    DATA_MODEL_XG.Model_Name = "XG Boosting"

    # # Building a Random Forest Model with adjusted parameters
    # def build_model_RF(
    #         Nb_Tree=1,
    #         min_samples_leaf=2,
    #         min_samples_split=10,
    #         max_depth=2,
    #         min_weight_fraction_leaf=0.5):

    #     MODEL_RF = RandomForestClassifier(
    #         n_estimators=Nb_Tree,
    #         min_samples_leaf=min_samples_leaf,
    #         min_samples_split=min_samples_split,
    #         random_state=0,
    #         max_depth=max_depth,
    #         min_weight_fraction_leaf=min_weight_fraction_leaf,)

    #     return MODEL_RF


    # # Searching for Optimized Hyperparameters
    # if Global_Parameters.RF_MODEL_OPTI:

    #     # Building function to minimize
    #     def objective_RF(trial):
    #         params = {'Nb_Tree': trial.suggest_int('Nb_Tree', 10, 250),
    #                   'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
    #                   'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
    #                   'max_depth': trial.suggest_int('max_depth', 1, 50),
    #                   'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)}

    #         MODEL_RF = build_model_RF(**params)
    #         scores = cross_val_score(
    #             MODEL_RF, DATA_MODEL_RF.X_train, DATA_MODEL_RF.Y_train, cv=Global_Parameters.k_folds)
    #         MODEL_RF.fit(DATA_MODEL_RF.X_train, DATA_MODEL_RF.Y_train)
    #         prediction_score = MODEL_RF.score(DATA_MODEL_RF.X_test, DATA_MODEL_RF.Y_test)

    #         return 0.1*np.amax(scores) + prediction_score

    #     # Search for best parameters
    #     study = optuna.create_study(direction='maximize')
    #     study.optimize(objective_RF, n_trials=Global_Parameters.RF_MODEL_TRIAL,
    #                    catch=(ValueError,))
    #     Best_params_RF = np.zeros([1], dtype=object)
    #     Best_params_RF[0] = study.best_params
    #     DATA_MODEL_RF.Nb_Tree = int(Best_params_RF[0].get("Nb_Tree"))
    #     DATA_MODEL_RF.min_samples_leaf = int(Best_params_RF[0].get("min_samples_leaf"))
    #     DATA_MODEL_RF.min_samples_split = int(Best_params_RF[0].get("min_samples_split"))
    #     DATA_MODEL_RF.min_weight_fraction_leaf = float(Best_params_RF[0].get("min_weight_fraction_leaf"))
    #     DATA_MODEL_RF.max_depth = int(Best_params_RF[0].get("max_depth"))

    DATA_MODEL_XG.XGBoosting_Modellisation(
        k_folds = Global_Parameters.k_folds)
    DATA_MODEL_XG.Evaluation_Metric_Plot_Mlogloss()
    DATA_MODEL_XG.Evaluation_Metric_Plot_Merror()
    DATA_MODEL_XG.Result_Plot()
    DATA_MODEL_XG.Feature_Importance_Plot()
    DATA_MODEL_XG.Result_Report()
    DATA_MODEL_XG.Result_Report_Plot()


# Saving model and information
joblib.dump(Global_Data.ARRAY_REPLACEMENT_ALL, "./data_replacement/array_replacement.joblib")
joblib.dump(Global_Data.INDEX_REPLACEMENT_ALL, "./data_replacement/index_replacement.joblib")

if Global_Parameters.RF_MODEL:
    with open('./models/rf_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_RF.MODEL, f)
elif Global_Parameters.NN_MODEL:
    with open('./models/nn_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_NN.MODEL, f)
elif Global_Parameters.GB_MODEL:
    with open('./models/gb_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_GB.MODEL, f)
elif Global_Parameters.XG_MODEL:
    with open('./models/xg_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_XG.MODEL, f)
