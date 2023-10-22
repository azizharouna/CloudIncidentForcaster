import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import category_encoders as ce

# Unzipping the provided dataset
with zipfile.ZipFile("incident+management+process+enriched+event+log.zip", 'r') as z:
    # Listing files in the zip archive
    file_names = z.namelist()
    # Loading the dataset (assuming the first file in the archive is the desired dataset)
    dataset_path = z.extract(file_names[0])

def custom_date_parser(date_str):
    try:
        # First, try the format "X-X-X H:M"
        return datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M")
    except ValueError:
        # If the above fails, try the format "X/X/X H:M"
        return datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M")
    

# write the custom parser to parse in addition dates of type "%d-%m-%Y %H:%M:%S" and "%d/%m/%Y %H:%M:%S" and "%d-%m-%Y %H:%M" and "%d/%m/%Y %H:%M"
def robust_date_parser(date_str):
    try:
        # First, try the format "X-X-X H:M:S"
        return datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")
    except ValueError:
        try:
            # If the above fails, try the format "X/X/X H:M:S"
            return datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            try:
                # If the above fails, try the format "X-X-X H:M"
                return datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M")
            except ValueError:
                # If the above fails, try the format "X/X/X H:M"
                return datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M")
            


class SmartEncoder:
    def __init__(self, 
                 one_hot_cols=[], 
                 label_cols=[], 
                 target_cols=[], 
                 binary_cols=[]):
        self.one_hot_cols = one_hot_cols
        self.label_cols = label_cols
        self.target_cols = target_cols
        self.binary_cols = binary_cols
        self.encoders = {}
    
    def fit(self, df, column, target_column=None):
        # If column is specified by the user
        if column in self.one_hot_cols:
            self.encoders[column] = 'one_hot'
        elif column in self.label_cols:
            self.encoders[column] = 'label'
        elif column in self.target_cols:
            if target_column is None:
                raise ValueError("Target column must be provided for target encoding.")
            encoder = ce.TargetEncoder()
            encoder.fit(df[column], df[target_column])
            self.encoders[column] = encoder
        elif column in self.binary_cols:
            encoder = ce.BinaryEncoder()
            encoder.fit(df[column])
            self.encoders[column] = encoder
        else:
            # Determine encoding method based on unique values
            unique_values = df[column].nunique()
            if unique_values <= 15:
                self.encoders[column] = 'one_hot'
            elif unique_values <= 100:
                self.encoders[column] = 'label'
            elif unique_values <= 1000:
                if target_column is None:
                    raise ValueError("Target column must be provided for target encoding.")
                encoder = ce.TargetEncoder()
                encoder.fit(df[column], df[target_column])
                self.encoders[column] = encoder
            else:
                encoder = ce.BinaryEncoder()
                encoder.fit(df[column])
                self.encoders[column] = encoder

    def transform(self, df, column):
        encoder = self.encoders.get(column)

        # Apply appropriate encoding based on the encoder determined during fit
        if encoder == 'one_hot':
            return pd.get_dummies(df, columns=[column], drop_first=True)
        
        elif encoder == 'label':
            df[column + '_encoded'] = df[column].astype('category').cat.codes
            return df.drop(columns=[column])
        
        elif encoder == 'high_cardinality':
            print(f"The column '{column}' has extremely high cardinality. Consider using advanced methods.")
            return df
        
        else:  # For target and binary encoders
            encoded_col = encoder.transform(df[column])
            if isinstance(encoded_col, pd.DataFrame):  # For binary encoder which returns multiple columns
                df = pd.concat([df, encoded_col], axis=1)
                return df.drop(columns=[column])
            else:  # For target encoder
                df[column + '_encoded'] = encoded_col
                return df.drop(columns=[column])

    def fit_transform(self, df, column, target_column=None):
        self.fit(df, column, target_column)
        return self.transform(df, column)

# Example Usage:
# encoder = SmartEncoder(one_hot_cols=['col1'], label_cols=['col2'], target_cols=['col3'])
# encoded_df = encoder.fit_transform(df, 'col_name', target_column='target')
        


