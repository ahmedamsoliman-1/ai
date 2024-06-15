import pandas as pd

class DataPreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe

    def preprocess(self):
        # Inspecting missing values
        missing_values = self.df.isnull().sum()
        print(f"Missing values before preprocessing:\n{missing_values}")

        # Handling nested fields and extracting features
        self.df['captured_piece_byblack_count'] = self.df['captured_piece_byblack'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        self.df['captured_piece_bywhite_count'] = self.df['captured_piece_bywhite'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # Dropping nested lists columns
        self.df.drop(columns=['captured_piece_byblack', 'captured_piece_bywhite'], inplace=True)

        # Handling missing values
        self.df['last_move_time_black'].fillna('Unknown', inplace=True)
        self.df['last_move_time_white'].fillna('Unknown', inplace=True)
        self.df['termination_id'].fillna('Unknown', inplace=True)
        self.df['game_end_time'].fillna('Unknown', inplace=True)
        self.df['termination_reason'].fillna('Unknown', inplace=True)
        self.df['winner'].fillna('Unknown', inplace=True)
        self.df['winner_id'].fillna('Unknown', inplace=True)

        # Dropping rows with critical missing values
        self.df.dropna(subset=['game_duration', 'total_move_duration_black', 'total_move_duration_white'], inplace=True)

        # Converting data types
        self.df['game_duration'] = self.df['game_duration'].astype(int)
        self.df['total_move_duration_black'] = self.df['total_move_duration_black'].astype(int)
        self.df['total_move_duration_white'] = self.df['total_move_duration_white'].astype(int)

        print(f"Data shape after preprocessing: {self.df.shape}")
        return self.df

    def get_numeric_data(self):
        # Select only numeric columns for modeling
        numeric_df = self.df.select_dtypes(include=[float, int])
        return numeric_df
