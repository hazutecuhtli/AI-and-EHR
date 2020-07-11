import pandas as pd
import numpy as np
import os, functools
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, auc, roc_curve

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    
    df = pd.merge(df, ndc_df[['NDC_Code','Non-proprietary Name']], left_on='ndc_code', right_on='NDC_Code', how='left')
    df = df.rename(columns={'Non-proprietary Name': 'generic_drug_name'})
    df.drop(['ndc_code', 'NDC_Code'], axis=1,inplace=True)
    
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    first_encounter_df = df.sort_values(['patient_nbr', 'encounter_id']).groupby('patient_nbr').first()  
    first_encounter_df.insert(1, 'patient_nbr', first_encounter_df.index)
    
    first_encounter_df.reset_index(drop=True, inplace=True)
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr', PREDICTOR_FIELD = 'time_in_hospital'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    X = df[[col for col in df.columns.tolist() if col != PREDICTOR_FIELD]]
    y = df[PREDICTOR_FIELD]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.5, random_state=42, stratify=y_test)   
    
    return X_train.join(y_train), X_val.join(y_val), X_test.join(y_test)

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key = c, 
                                                                                                  vocabulary_file = vocab_file_path,
                                                                                                  num_oov_buckets=1) 
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)        
                
        
        output_tf_list.append(tf_categorical_feature_column)
        
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    normalizer = functools.partial(normalize_numeric_with_zscore, mean = MEAN, std = STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key = col,
                                                          default_value = default_value, 
                                                          normalizer_fn = normalizer, 
                                                          dtype = tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    
    cond = 5 #to 7
    
    student_binary_prediction = df.apply(lambda x: 1 if x[col] >= cond else 0, axis=1).values
        
    return student_binary_prediction

def pred_performance_summary(df):
    
    fpr, tpr, _ = roc_curve(df.score, df.label_value)

    summary = {}

    idxs = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'auc']
    values = [accuracy_score(df.score, df.label_value),
              precision_score(df.score, df.label_value),
              recall_score(df.score, df.label_value),
              f1_score(df.score, df.label_value),
              auc(fpr, tpr)]


    display(pd.DataFrame(data=values, index=idxs, columns=['values']))

    plt.figure(figsize=(10,7))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % values[-1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def demographics(df):

    def percentages(x):

        suma = x.sum(axis=1).tolist()
        for i, idx in enumerate(x.index):
            x.loc[idx] = x.loc[idx]/suma[i]

        return x

    #d. Please describe the demographic distributions in the dataset for the age and gender fields.
    fig = plt.figure(constrained_layout=True, figsize=(15,10));

    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :-1])
    cmap = plt.cm.get_cmap('tab10', 10)
    sns.countplot(x='age',data=df, ax=ax1)
    ax1.set_title('Participants per group of ages')
    ax1.set_ylabel("Number of participants")   
    ax1.set_xlabel(" ")    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)    
    ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[0, 2])
    df.race.value_counts().plot.bar(width=0.7, ax=ax2)
    ax2.set_title('Participants per race')
    ax2.set_ylabel("Number of participants")  
    ax2.grid(True)    
    
    ax3 = fig.add_subplot(gs[1, :-1])
    df.groupby(["age"]).gender.value_counts().unstack().apply(lambda x: x/(x['Female'] + x['Male']), axis=1).plot.bar(width=0.7, ax=ax3)
    ax3.grid(True)       
    ax3.set_title('Genders per Race')
    ax3.set_ylabel("Percentages for the pupilation of races")
    ax3.set_xlabel(" ")        
    
    ax4 = fig.add_subplot(gs[1, 2])
    sns.countplot(x='gender',data=df, ax=ax4)
    ax4.set_title('Participants per gender')
    ax4.set_ylabel("Number of participants")  
    ax4.set_xlabel(" ")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)
    ax4.grid(True)
    