from utils import StreamLogger
stream_logger = StreamLogger()

from src import DataFetcher
from src import DataPreprocessor
from src import EDA
from src import Model
from src import Visualize
from _cred import HOST, USERNAME, PASSWORD, CA_BUNDLE_PATH

INDEX_NAME = "cs2_datapipeline_alpha_cs2_games"

stream_logger.stream_logger.system('Starting EDA and Model Training')

# Fetch data
data_fetcher = DataFetcher(HOST, USERNAME, PASSWORD, CA_BUNDLE_PATH)
query = {
    "query": {
        "match_all": {}
    }
}
data = data_fetcher.fetch_data(INDEX_NAME, query)
stream_logger.stream_logger.system("Fetched data:")
stream_logger.stream_logger.system(data.head())

# Preprocess data
preprocessor = DataPreprocessor(data)
clean_data = preprocessor.preprocess()
numeric_data = preprocessor.get_numeric_data()  # Get numeric columns only
stream_logger.stream_logger.system("Clean data after preprocessing:")
stream_logger.stream_logger.system(clean_data.head())

if numeric_data.empty:
    stream_logger.stream_logger.warning("Clean data is empty after preprocessing. Please check the preprocessing steps.")
else:
    # Perform EDA
    stream_logger.stream_logger.warning("Performing EDA...")
    eda = EDA(clean_data)
    eda.visualize_data()
    stream_logger.stream_logger.warning("EDA Completed.")

    # Train and evaluate model
    stream_logger.stream_logger.warning("Training and evaluating model...")
    target_column = "winner"  # Predicting the winner
    model = Model(numeric_data, clean_data[target_column])  # Pass numeric data and target
    evaluation_report = model.train_and_evaluate()
    stream_logger.stream_logger.system("Model evaluation report:")
    stream_logger.stream_logger.system(evaluation_report)

    # Visualize feature importance
    stream_logger.stream_logger.warning("Visualizing feature importance...")
    feature_importances = model.feature_importance()
    visualize = Visualize(feature_importances)
    visualize.plot_feature_importance()
    stream_logger.stream_logger.warning("Feature importance visualization completed.")
