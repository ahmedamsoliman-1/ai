from elasticsearch import Elasticsearch
import pandas as pd

class DataFetcher:
    def __init__(self, host, username, password, ca_bundle_path):
        self.es = Elasticsearch(
            [host],
            basic_auth=(username, password),
            ca_certs=ca_bundle_path,
            # ssl_context=self.create_ssl_context(ca_bundle_path)
        )

    @staticmethod
    def create_ssl_context(ca_bundle_path):
        import ssl
        context = ssl.create_default_context(cafile=ca_bundle_path)
        return context
    
    def fetch_data(self, index_name, query):
        response = self.es.search(index=index_name, body=query)
        data = [hit["_source"] for hit in response["hits"]["hits"]]
        return pd.DataFrame(data)
