import pandas as pd
import warnings # DO NOT modify this line
from sklearn.exceptions import ConvergenceWarning # DO NOT modify this line
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=ConvergenceWarning) # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path): # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=',')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self): # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        # TODO: Paste your code here
        return self.df.shape[0]

    def Q2(self): # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        return (self.df.select_dtypes(include=['number']).shape[1], self.df.select_dtypes(include=['object','category']).shape[1])
    
    def Q3(self): # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        return (float(round(self.df['y'].value_counts(normalize=True).get('no', 0), 3)), float(round(self.df['y'].value_counts(normalize=True).get('yes', 0), 3)))
      
    

    def Q4(self): # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        return self.df.drop_duplicates().shape
        

    def Q5(self): # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values. 
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        self.df = self.df.drop_duplicates()

        self.df = self.df.replace('unknown', pd.NA)
        cols = self.df.columns
        for col in cols:
            if self.df[col].value_counts(normalize=True).iloc[0] > 0.99:
                self.df.drop([col], axis=1, inplace=True)
        
        X = self.df.drop(columns=['y'], axis=1)
        y = self.df['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,stratify=y,random_state=0,test_size=0.3)

        return (self.X_train.shape, self.X_test.shape)

       
    def Q6(self): 
        """
        Problem 6: 
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the nominal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7} 
                Hint: Use One hot encoder or pd.dummy to encode nominal category
            return the shape of X_train.

        """
        df_tmp = self.df.copy()

        df_tmp = df_tmp.drop_duplicates()

        df_tmp = df_tmp.replace('unknown', pd.NA)
        cols = df_tmp.columns
        for col in cols:
            if df_tmp[col].value_counts(normalize=True).iloc[0] > 0.99:
                df_tmp.drop([col], axis=1, inplace=True)
        
        X = df_tmp.drop(columns=['y'], axis=1)
        y = df_tmp['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,stratify=y,random_state=0,test_size=0.3)

        # Impute missing values
        for col in self.X_train.select_dtypes(include=['number']).columns:
            mean_val = self.X_train[col].mean()  # only form train set
            self.X_train[col] = self.X_train[col].fillna(mean_val)
            self.X_test[col] = self.X_test[col].fillna(mean_val)

        for col in self.X_train.select_dtypes(include=['object','category']).columns:
            mode_val = self.X_train[col].mode().iloc[0]
            self.X_train[col] = self.X_train[col].fillna(mode_val)
            self.X_test[col] = self.X_test[col].fillna(mode_val)

        # Encode education variable
        education_order = {
            'illiterate': 1,
            'basic.4y': 2,
            'basic.6y': 3,
            'basic.9y': 4,
            'high.school': 5,
            'professional.course': 6,
            'university.degree': 7
        }
        self.X_train['education'] = self.X_train['education'].map(education_order)
        self.X_test['education'] = self.X_test['education'].map(education_order)    

        self.X_train = pd.get_dummies(self.X_train)
        self.X_test = pd.get_dummies(self.X_test)

        return self.X_train.shape


    def Q7(self):
        ''' Problem7: Use Logistic Regression as the model with 
            random_state=2025, 
            class_weight='balanced' and 
            max_iter=500. 
            Train the model using all the remaining available variables. 
            What is the macro F1 score of the model on the test data? in 3 digits
        '''
        df_tmp = self.df.copy()

        df_tmp = df_tmp.drop_duplicates()

        df_tmp = df_tmp.replace('unknown', pd.NA)
        cols = df_tmp.columns
        for col in cols:
            if df_tmp[col].value_counts(normalize=True).iloc[0] > 0.99:
                df_tmp.drop([col], axis=1, inplace=True)
        
        X = df_tmp.drop(columns=['y'], axis=1)
        y = df_tmp['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,stratify=y,random_state=0,test_size=0.3)

        # Impute missing values
        for col in self.X_train.select_dtypes(include=['number']).columns:
            mean_val = self.X_train[col].mean()  # only form train set
            self.X_train[col] = self.X_train[col].fillna(mean_val)
            self.X_test[col] = self.X_test[col].fillna(mean_val)

        for col in self.X_train.select_dtypes(include=['object','category']).columns:
            mode_val = self.X_train[col].mode().iloc[0]
            self.X_train[col] = self.X_train[col].fillna(mode_val)
            self.X_test[col] = self.X_test[col].fillna(mode_val)

        # Encode education variable
        education_order = {
            'illiterate': 1,
            'basic.4y': 2,
            'basic.6y': 3,
            'basic.9y': 4,
            'high.school': 5,
            'professional.course': 6,
            'university.degree': 7
        }
        self.X_train['education'] = self.X_train['education'].map(education_order)
        self.X_test['education'] = self.X_test['education'].map(education_order)    

        self.X_train = pd.get_dummies(self.X_train)
        self.X_test = pd.get_dummies(self.X_test)

        model = LogisticRegression(random_state=2025, class_weight='balanced', max_iter=500)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        return round(report['macro avg']['f1-score'],2)
