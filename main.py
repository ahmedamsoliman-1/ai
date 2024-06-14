from utils import StreamLogger


stream_logger = StreamLogger()

from src import DataFetcher
from src import DataPreprocessor
from src import EDA
from src import Model
from src import Visualize


# Elasticsearch connection settings
HOST = "https://ai-es02-dev.avrioc.io:9200"
USERNAME = "SVC-analytics"
PASSWORD = "SVCAnalytics456#4"
CA_BUNDLE_PATH = "/Users/ahmed.soliman/workspace/cs2/certs/avrioc.iobundle.crt"

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

stream_logger.stream_logger.warning(data)
# # Preprocess data
# preprocessor = DataPreprocessor(data)
# clean_data = preprocessor.preprocess()
# stream_logger.stream_logger.system(clean_data)

# # Perform EDA
# eda = EDA(clean_data)
# eda.visualize_data()

# # Train and evaluate model
# target_column = "target_column"  # Replace with your target column
# model = Model(clean_data, target_column)
# print(model.train_and_evaluate())

# # Visualize feature importance
# feature_importances = model.feature_importance()
# visualize = Visualize(feature_importances)
# visualize.plot_feature_importance()
