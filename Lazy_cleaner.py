import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as sm

from scipy import stats

import chardet

import fuzzywuzzy
from fuzzywuzzy import process

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



def fill_lin_rand(messy_df, metric, colnames):
        """
        messy_df -> dataframe contain Nans
        metric -> weather fill Nans by linear regression or random forest
        colnames ->column names that needs to replace the Nans with numeric values
        
        this function is to fill the Nan values of columns by making 
        a regression modle by taking features of Non-Nan values as a
        training data and predicting the missing values and fill it
        
        returns:
        the clean dataframe and list of the missing values
        """

        # Create X_df of predictor columns
        X_df = messy_df.drop(colnames, axis = 1)

        # Create Y_df of predicted columns
        Y_df = messy_df[colnames]

        # Create empty dataframes and list
        Y_pred_df = pd.DataFrame(columns=colnames)
        Y_missing_df = pd.DataFrame(columns=colnames)
        missing_list = []

        # Loop through all columns containing missing values
        for col in messy_df[colnames]:

            # Number of missing values in the column
            missing_count = messy_df[col].isnull().sum()

            # Separate train dataset which does not contain missing values
            messy_df_train = messy_df[~messy_df[col].isnull()]

            # Create X and Y within train dataset
            msg_cols_train_df = messy_df_train[col]
            messy_df_train = messy_df_train.drop(colnames, axis = 1)

            # Create test dataset, containing missing values in Y    
            messy_df_test = messy_df[messy_df[col].isnull()]

            # Separate X and Y in test dataset
            msg_cols_test_df = messy_df_test[col]
            messy_df_test = messy_df_test.drop(colnames,axis = 1)

            # Copy X_train and Y_train
            Y_train = msg_cols_train_df.copy()
            X_train = messy_df_train.copy()

            # Linear Regression model
            if metric == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train,Y_train)
                print("R-squared value is: " + str(model.score(X_train, Y_train)))

            # Random Forests regression model
            elif metric == "Random Forests":
                model = RandomForestRegressor(n_estimators = 10 , oob_score = True)
                model.fit(X_train,Y_train) 

    #             importances = model.feature_importances_
    #             indices = np.argsort(importances)
    #             features = X_train.columns

    #             print("Missing values in"+ col)
    #             #plt.title('Feature Importances')
    #             plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    #             plt.yticks(range(len(indices)), features) ## removed [indices]
    #             plt.xlabel('Relative Importance')
    #             plt.show()

            X_test = messy_df_test.copy()

            # Predict Y_test values by passing X_test as input to the model
            Y_test = model.predict(X_test)

            Y_test_integer = pd.to_numeric(pd.Series(Y_test),downcast='integer')

            # Append predicted Y values to known Y values
            Y_complete = Y_train.append(Y_test_integer)
            Y_complete = Y_complete.reset_index(drop = True)

            # Update list of missing values
            missing_list.append(Y_test.tolist())

            Y_pred_df[col] = Y_complete
            Y_pred_df = Y_pred_df.reset_index(drop = True)

        # Create cleaned up dataframe
        clean_df = X_df.join(Y_pred_df)

        return clean_df,missing_list



class stat():
    def __init__(self):
        pass
        
    def p_val_of_features(self,data,target_column):
    
        """
        df -> the dataframe
        label_column -> the column you want to predict
        
        this function calculate the P value of the features to know how it affects the regression module
        for a single label
        
        returns:
        summary report
        """
        # stre is a string variable of independent and dependant columns

        stre = '{} ~'.format(target_column) 

        for i in data.columns:
            stre = stre + "{} +".format(i)

        stre = stre[0:-1]  #to remove the last + sign

        reg_ols = sm.ols(formula=stre, data=data).fit() 

        return reg_ols.summary()
    
#------------------------------------------------------------------------------------------------------------------------------
     
    def na_per_data(self,data):
        """
        df -> dataframe
        
        returns:
        the percentage of Nan values in the whole dataset
        """
        per = (((data.isnull().sum()).sum())/ np.product(data.shape))*100
        return per



class fillnan():
    def __init__(self):
        pass
    
    def simpleimputer(self,df):
        """
        df -> dataframe
        
        this function fill the nan values with simple techniques like (mean,mode) 
        depending on the data type of the column
        
        *also consider filling by median manually before using this method*
        
        returns:
        the dataframe after editing
        """
        for i in df.columns:
            if df[i].isnull().any() == True :
                if df[i].dtypes == "object":
                    if len(df[i].unique()) <= 10:
                        df[i].fillna(df[i].mode()[0],inplace=True)

                    if len(df[i].unique()) > 10 :
                        df[i].dropna(inplace=True)

                if df[i].dtypes == "int64" or df[i].dtypes == "int32" or df[i].dtypes == "float64":

                    if len(df[i].unique()) <= 10 :
                        df[i].fillna(df[i].mode()[0],inplace=True)

                    if len(df[i].unique()) > 10 :
                        df[i].fillna(df[i].mean(),inplace=True)

                else:
                    df[i].dropna(inplace=True) 

        return df
    
 #------------------------------------------------------------------------------------------------------------------------------
    def hyperimputer(self,df,metric = "Linear Regression"): # there is also "Random Forests"
        """
        df->dataframe
        metric ->"Linear Regression" or "Random Forests" models to fill you numeric nan values with

        this function compines between the simple imputation and the linear regression imputation
        as in object columns it will impute with the mode and for any numerical number it will impute 
        with linear regression or random forest

        *also consider filling by median manually before using this method*

        returns:
        the dataframe after editing
        """
        for i in df.columns:
            if df[i].isnull().any() == True :
                if df[i].dtypes == "object":

                    if len(df[i].unique()) <= 10:
                        df[i].fillna(df[i].mode()[0],inplace=True)

                    if len(df[i].unique()) > 10 :
                        df[i].dropna(inplace=True)

                if df[i].dtypes == "int64" or df[i].dtypes == "int32" or df[i].dtypes == "float64":

                    if len(df[i].unique()) > 10:
                        df,_ = fill_lin_rand(df,metric,[i])

                else:
                    df[i].dropna(inplace=True) 

        return df

 #------------------------------------------------------------------------------------------------------------------------------

    def fill_by_nex(self,df,columns=[]):
        """
        df -> dataframe
        columns -> a list of columns you want to fill
        
        this one fill nan values by the next value

        returns:
        the dataframe after editing
        """
        for i in columns:
            df[i] = df[i].fillna(method='bfill', axis=0).fillna(0)
            
        return df
    
 #------------------------------------------------------------------------------------------------------------------------------
    
    def fill_by_perv(self,df,columns=[]):
        """
        df -> dataframe
        columns -> a list of columns you want to fill
        
        this one fill nan values by the previous value

        returns:
        the dataframe after editing
        """
        for i in columns:
            df[i] = df[i].fillna(method='ffill', axis=0).fillna(0)
            
        return df



class label():
    def __init__(self):
        pass
    
    def to_category(self,df):
        """
        change from and object datatype column into categorie datatype column
        
        df-> dataframe
        
        returns:
        the dataframe after editing
        """
        cols = df.select_dtypes(include='object').columns
        for col in cols:
            ratio = len(df[col].value_counts()) / len(df)
            if ratio < 0.05:
                df[col] = df[col].astype('category')
        return df
    
 #------------------------------------------------------------------------------------------------------------------------------

    def freq_labeling(self,df=None,column=None):
        """
        replace objects by how frequent they are in a certain column
        
        df -> dataframe
        column -> column you want to apply this method on
        
        returns:
        the dataframe after editing
        """
        df = df.copy()
        freq = (df[column].value_counts() /len(df))
        d={}

        for i in freq.index:
            d[i] = freq[i]

        df[column] = df[column].map(d)

        return df



class Clean_Data():
    def __init__(self):
        pass
    
    def reduce_mem_usage(self,props):

        """
        this funaction to reduce memory usage of dataset
        props-> dataset you want to reduce
        
        returns:
        the dataframe after editing and Nan values list
        """

        start_mem_usg = props.memory_usage().sum() / 1024**2
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
        NAlist = [] # Keeps track of columns that have missing values filled in. 
        for col in props.columns:
            if props[col].dtype not in [object, bool]:  # Exclude strings

                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",props[col].dtype)

                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()
                '''
                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(props[col]).all(): 
                    NAlist.append(col)
                    props[col].fillna(mn-1,inplace=True) 
                '''

                # test if column can be converted to an integer
                asint = props[col].fillna(0).astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True



                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            props[col] = props[col].astype(np.uint8)
                        elif mx < 65535:
                            props[col] = props[col].astype(np.uint16)
                        elif mx < 4294967295:
                            props[col] = props[col].astype(np.uint32)
                        else:
                            props[col] = props[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props[col] = props[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props[col] = props[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props[col] = props[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props[col] = props[col].astype(np.int64)    

                # Make float datatypes 32 bit
                else:
                    props[col] = props[col].astype(np.float32)

                # Print new column type
                print("dtype after: ",props[col].dtype)
                print("******************************")

        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
        return props, NAlist
    
#-----------------------------------------------------------------------------------------------------------------------------

    def replace_matches(self,df, column, string_to_match, min_ratio = 47):
        """
        if there is simillar words but different format due to a data entry error
        you can apply this function to calculate similarity percentage and then change them
        
        df -> dataframe
        column -> column to edit
        string_to_match -> string you want to match and replace
        min_ratio -> minimum probability to excange
        
        returns:
        the dataframe after editing
        """
        # get a list of unique strings
        strings = df[column].unique()

        # get the top 10 closest matches to our input string
        matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                             limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

        # only get matches with a ratio > 90
        close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

        # get the rows of all the close matches in our dataframe
        rows_with_matches = df[column].isin(close_matches)

        # replace all rows with close matches with the input matches 
        df.loc[rows_with_matches, column] = string_to_match

        # let us know the function's done
        print("All done!")
        
        return df

#-----------------------------------------------------------------------------------------------------------------------------
    def drop_missing(self,df,thresh=55):
        """
        drop the columns if the missing values exceed 60% of it
        
        df-> dataframe
        thresh->percentage of the missing threshold to Delete above
        
        returns:
        the dataframe after editing
        """
        thresh = len(df) * (thresh/100)
        df.dropna(axis=1, thresh=thresh, inplace=True)
        return df
    
#------------------------------------------------------------------------------------------------------------------------------
        
    def log_features(self,df=None):
    
        """
        log the data to remove large gaps between the data
        after or before removing outliers
        
        df -> dataframe you want to apply log function to
        
        returns:
        dataset after applying log 
        """
        if 0 in df.values:
            df= np.log1p(df)
        
        if 0 not in df.values:
            df= np.log(df)
            df[df == -inf] = 0

        return df
    
#------------------------------------------------------------------------------------------------------------------------------

    def dealing_with_outliers(self,df , type_o = "z-score"):
    
        """
        this function deals and removes outliers with z-score and Inter-Quartile Range
        method
        hint : XGboost deal with it very good (SOTA machine learning model)
        df -> dataframe
        type_o -> type of the method you want to choose
        
        returns:
        the dataframe after editing
        """

        if type_o == "z-score":

            # z-score range.

            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]

        if type_o == "IQR" :

            #Inter-Quartile Range

            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


        return df
    
#-----------------------------------------------------------------------------------------------------------------------------------
    
    def normalize(self,df,columns=[]):
        """
        normalize columns by your choice in a dataframe

        df -> dataframe
        columns -> list of columns to normalize its values

        returns:
        the dataframe after editing
        """
        for i in columns:
            df[i] = stats.boxcox(df[i])
                
        return df
#----------------------------------------------------------------------------------------------------------------------------------

    def en_de(self,decode=False,typeo="utf-8"):
        """
        encode and decode any string format
        
        decode -> if you want to encode assign as False if decode assign True
        typeo -> the format you want to encode to or decode from
        
        returns:
        the result after encoding or decoding
        """
        #or ascii
        if decode == True :
            return bite.decode(typeo)
        else :
            return string.encode(typeo, errors="replace")

