import pandas as pd
from sklearn.model_selection import train_test_split

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)

"""


def Q1(df):
    """
        Problem 1:
            How many rows are there in the "titanic_to_student.csv"?
    """
    
    return df.shape[0]


def Q2(df):
    '''
        Problem 2:
            2.1 Drop variables with missing > 50%
            2.2 Check all columns except 'Age' and 'Fare' for flat values, drop the columns where flat value > 70%
            From 2.1 and 2.2, how many columns do we have left?
            Note: 
            -Ensure missing values are considered in your calculation. If you use normalize in .value_counts(), please include dropna=False.
    '''
    half_count = len(df) / 2

    new_df = df.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values

    for col in new_df.columns:
        if col not in ['Age','Fare']:
            top_freq = new_df[col].value_counts(dropna=False, normalize=True).values[0]  # find the most frequent value's frequency
            if top_freq > 0.7:
                new_df = new_df.drop(col, axis=1)

    return new_df.shape[1]


def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    # subset means columns to consider for identifying missing values
    return df.dropna(subset=['Survived']).shape[0] 


def Q4(df):
    '''
       Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    df_copy = df.copy()
    
    Q1 = df_copy['Fare'].quantile(0.25)
    Q3 = df_copy['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_copy['Fare'] = df_copy['Fare'].apply(
        lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
    )
    
    return round(df_copy['Fare'].mean(), 2)


def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    df_copy = df.copy()
    mean_age = df_copy['Age'].mean()
    df_copy['Age'] = df_copy['Age'].fillna(mean_age)
    return round(df_copy['Age'].mean(), 2)


def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # 1. คัดลอก DataFrame เพื่อป้องกันการแก้ไขข้อมูลต้นฉบับ
    df_copy = df.copy()

    # 2. ทำ Dummy Coding (จะสร้างคอลัมน์ใหม่ เช่น Embarked_C, Embarked_Q, Embarked_S)
    df_dummy = pd.get_dummies(df_copy, columns=['Embarked'])

    # 3. คำนวณค่าเฉลี่ยของคอลัมน์ "Embarked_Q"
    ans = df_dummy['Embarked_Q'].mean()

    # 4. ส่งคืนค่าที่ปัดเศษ 2 ตำแหน่ง
    return round(ans, 2)


def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection, 
            Don't forget to impute missing values with mean.
    '''

    df_prep = df.copy()
    df_prep = df_prep.fillna(df_prep.select_dtypes(include='number').mean())
    y = df_prep.pop('Survived')
    X = df_prep

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3, random_state=123)
    # count of survived (1) / total count in y_train
    return round(y_train.value_counts()[1]/y_train.shape[0], 2)