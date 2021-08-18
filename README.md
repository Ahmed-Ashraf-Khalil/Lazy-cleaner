# Lazy-cleaner:

![lazy](https://user-images.githubusercontent.com/59618586/128797360-b96bb2e6-f7f8-42b5-aed0-a38fbfd37811.png)

## What it is about :
***Lazy-cleaner*** is a python package that cleans your numeric data and that includes correcting data types and class labeling techniques and filling the Null values with more than one method saving the time of coding exeplicit methods.

<br/>

## Advanteges
1. saving time.
2. preprocessing data like a pro.
3. user friendly for begineer Data learners.

## Installation Guide
To install ***Lazy-cleaner*** Package:
- Make sure python3 and pip is installed in your machine.
- Use the command line and type: `pip install Lazy-cleaner` or `pip3 install Lazy-cleaner`

## How to Use?
The most simple format of using ***Lazy-cleaner*** is as follow:
```python
from Lazy_cleaner import stat
from Lazy_cleaner import fillnan
from Lazy_cleaner import label
from Lazy_cleaner import Clean_Data

import pandas as pd

Data = pd.read_csv("FileName.csv")
fill = fillnan()

New_Data = fill.simpleimputer(Data)
```
## Existing functions:

#### Lazy_cleaner.fill_lin_rand( messy_df, metric , colnames ): 
fill the Nan values by Linear Regression or by Random Forest.

*also consider filling by median manually before using this method*

* messy_df -> data frame you want to work on.
* metric -> model you want to work with (Linear Reg , Random forest).
* colnames-> list of column names that contains Null vlaues.

Returns : the cleaned dataframe after filling the Null values and a list of the missing values

<hr/>

#### Lazy_cleaner.stat().p_val_of_features( df , label_column ): 
calculate Ordinary Least Squares to know which features are important and make overview of the regression model.

* df -> the dataframe.
* label_column -> the Target column Name in string format.

Returns : summary report about the features and their p value

<br/>

#### Lazy_cleaner.stat().na_per_data( df ):
calculate the percentage of Nan values in the whole dataset

* df -> dataframe

returns:
the percentage of Nan values in the whole dataset

<hr/>

#### Lazy_cleaner.fillnan().simpleimputer( df ):

this function fill the Null values with simple techniques like (mean,mode) 
depending on the data type of the column

*also consider filling by median manually before using this method*

* df -> dataframe

returns:
the dataframe after editing

<br/>

#### Lazy_cleaner.fillnan().hyperimputer( df , metric = "Linear Regression" ):
this function compines between the simple imputation , the linear regression imputation and random forests imputation as in object columns it will impute with the mode and for any numerical number it will impute with linear regression or random forest

*also consider filling by median manually before using this method*

* df->dataframe
* metric ->"Linear Regression" or "Random Forests" models to fill you numeric nan values with

returns:
the dataframe after editing

<br/>

#### Lazy_cleaner.fillnan().fill_by_nex(df,columns=[]):
fill nan values by the next value

* df -> dataframe
* columns -> a list of columns you want to fill
        

returns: 
the dataframe after editing
        
<br/>

#### Lazy_cleaner.fillnan().fill_by_perv(df,columns=[]):
this one fill nan values by the previous value

* df -> dataframe
* columns -> a list of columns you want to fill
        
returns:
the dataframe after editing

<hr/>

#### Lazy_cleaner.label().to_category( df ):
change from and object datatype column into categorie datatype column

* df-> dataframe

returns:
the dataframe after editing

<br/>

#### Lazy_cleaner.label().freq_labeling( df , column ):
replace objects by how frequent they are in a certain column

* df -> dataframe
* column -> column you want to apply this method on

returns:
the dataframe after editing

<hr/>

#### Lazy_cleaner.Clean_Data().reduce_mem_usage( props ):
reduce memory usage of the dataframe.

* props-> dataset you want to reduce

returns:
the dataframe after editing and Nan values list

<br/>

#### Lazy_cleaner.Clean_Data().replace_matches( df , column , string_to_match , min_ratio ):
if there is simillar words but different format due to a data entry 
you can apply this function to calculate similarity percentage and then change them

* df -> dataframe
* column -> column to edit
* string_to_match -> string you want to match and replace
* min_ratio -> minimum probability to excange

returns:
the dataframe after editing

<br/>

#### Lazy_cleaner.Clean_Data().drop_missing( df , thresh=55 ):
drop the columns if the missing values exceed 60% of it

* df-> dataframe
* thresh->percentage of the missing threshold to Delete above

returns:
the dataframe after editing

<br/>

#### Lazy_cleaner.Clean_Data().log_features( df=None ):
apply Log function to the features.

* df -> dataframe you want to apply log func to

returns:
dataset after applying log 

<br/>

#### Lazy_cleaner.Clean_Data().dealing_with_outliers( df , type_o = "z-score"):
removes outliers with two techniques z-score and IQR ("Inter-Quartile Range"). 

* df -> dataframe
* type_o -> type of the method you want to choose

Returns : the dataframe after removing the outliers

<br/>

#### Lazy_cleaner.Clean_Data().normalize( df , columns ):
normalize columns by your choice in a dataframe

* df -> dataframe
* columns -> list of columns to normalize its values

returns:
the dataframe after editing

<br/>

#### Lazy_cleaner.Clean_Data().en_de( decode , typeo ):
encode and decode any string format

* decode -> if you want to encode assign as False if decode assign True
* typeo -> the format you want to encode to or decode from

returns:
the result after encoding or decoding


