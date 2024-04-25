import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


class EllipticDataset:
    def __init__(self, config):
        self.features_df = pd.read_csv(config.features_path, header=None)
        self.edges_df = pd.read_csv(config.edges_path)
        self.labels_df = pd.read_csv(config.classes)
        self.labels_df["class"] = self.labels_df["class"].map(
            {"unknown": 2, "1": 1, "2": 0}
        )
        self.merged_df = self.merge()
        self.edge_index = self._edge_index()
        self.edge_weights = self._edge_weights()
        self.node_features = self._node_features()
        self.labels = self._labels()
        self.classified_ids = self._classified_ids()
        self.unclassified_ids = self._unclassified_ids()
        self.licit_ids = self._licit_ids()
        self.illicit_ids = self._illicit_ids()

    def visualize_distribution(self):
        groups = self.labels_df.groupby("class").count()
        plt.title("Classes distribution")
        plt.barh(
            ["Licit", "Illicit", "Unknown"],
            groups["txId"].values,
            color=["green", "red", "grey"],
        )

    def merge(self):
        df_merge = self.features_df.merge(
            self.labels_df, how="left", right_on="txId", left_on=0
        )
        df_merge = df_merge.sort_values(0).reset_index(drop=True)
        return df_merge

    def train_test_split(self, test_size=0.15):
        train_idx, valid_idx = train_test_split(
            self.classified_ids.values, test_size=test_size
        )
        return train_idx, valid_idx

    def pyg_dataset(self):
        dataset = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights,
            y=self.labels,
        )
        train_idx, valid_idx = self.train_test_split()
        dataset.train_idx = train_idx
        dataset.valid_idx = valid_idx
        dataset.test_idx = self.unclassified_ids

        return dataset

    def _licit_ids(self):
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        licit_ids = node_features["class"].loc[node_features["class"] == 0].index
        return licit_ids

    def _illicit_ids(self):
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        illicit_ids = node_features["class"].loc[node_features["class"] == 1].index
        return illicit_ids

    def _classified_ids(self):
        """
        Get the list of labeled ids
        """
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        unclassified_ids = node_features["class"].loc[node_features["class"] != 2].index
        return unclassified_ids

    def _unclassified_ids(self):
        """
        Get the list of unlabeled ids
        """
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        unclassified_ids = node_features["class"].loc[node_features["class"] == 2].index
        return unclassified_ids

    def _node_features(self):
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        node_features = node_features.drop(columns=[0, 1, "class"])
        node_features_t = torch.tensor(node_features.values, dtype=torch.double)

        return node_features_t

    def _edge_index(self):
        node_ids = self.merged_df[0].values
        ids_mapping = {y: x for x, y in enumerate(node_ids)}
        edges = self.edges_df.copy()
        edges.txId1 = edges.txId1.map(
            ids_mapping
        )  # get nodes idx1 from edges_df list and filtered data
        edges.txId2 = edges.txId2.map(ids_mapping)
        edges = edges.astype(int)

        edge_index = np.array(edges.values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        return edge_index

    def _edge_weights(self):
        weights = torch.tensor([1] * self.edge_index.shape[1], dtype=torch.double)
        return weights

    def _labels(self):
        labels = self.merged_df["class"].values
        labels_tensor = torch.tensor(labels, dtype=torch.double)
        return labels_tensor


class FraudDataset:
    def __init__(
        self,
        config,
    ):
        # Load training data
        self.train_df = self.load_and_merge_data(
            config.train_transaction_path, config.train_identity_path, config.chunksize
        )
        self.test_df = self.load_and_merge_data(
            config.test_transaction_path, config.test_identity_path, config.chunksize
        )
        print("Data loaded")

        # Encode categorical variables
        self._encode_categoricals()

        # Create tensors for PyG data object
        self.features = self._node_features(self.train_df)
        self.labels = self._labels(self.train_df)
        self.test_features = self._node_features(self.test_df)
        print("Tensors created")

        self.edge_index = self._edge_index()

    def load_and_merge_data(self, transaction_path, identity_path, chunksize=None):
        print(f"Loading from {transaction_path} with chunksize : {chunksize}")
        if chunksize is None:
            # Load data normally
            transaction_df = pd.read_csv(transaction_path)
            identity_df = pd.read_csv(identity_path)
        else:
            # Load data in chunks
            transaction_df = pd.concat(
                pd.read_csv(transaction_path, chunksize=chunksize)
            )
            identity_df = pd.concat(pd.read_csv(identity_path, chunksize=chunksize))

        # Merge datasets
        print(f"Loading Done, Merging..")
        df = pd.merge(transaction_df, identity_df, on="TransactionID", how="left")
        print("Merging Done")
        return df

    def _encode_categoricals(self):
        # Combine train and test to ensure consistent encoding
        combined = pd.concat([self.train_df, self.test_df])
        categoricals = combined.select_dtypes(include=["object"]).columns.tolist()
        combined_encoded = pd.get_dummies(combined, columns=categoricals)

        # Split combined back into train and test
        self.train_df = combined_encoded.iloc[: len(self.train_df)]
        self.test_df = combined_encoded.iloc[len(self.train_df) :]

    def _node_features(self, df):
        # Drop target and non-feature columns, then convert to tensor
        features = df.drop(["isFraud", "TransactionID"], axis=1)
        return torch.tensor(features.values, dtype=torch.float)

    def _labels(self, df):
        # Convert labels to tensor
        return torch.tensor(df["isFraud"].values, dtype=torch.long)

    def _encode_nodes(self):
        # Assuming 'TransactionID' is the primary key for transactions
        # All other ID columns are considered as separate nodes
        id_cols = [
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "ProductCD",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
        ]
        node_dict = {}
        index = 0

        # Assign an index to each unique transaction
        self.train_df["node_index"] = range(len(self.train_df))
        node_dict["Transaction"] = dict(
            zip(self.train_df["TransactionID"], self.train_df["node_index"])
        )

        # Assign indices to other identifiers
        for col in id_cols:
            unique_values = pd.unique(self.train_df[col].dropna())
            node_dict[col] = {value: i + index for i, value in enumerate(unique_values)}
            index += len(unique_values)

        return node_dict

    def _edge_index(self):
        node_dict = self._encode_nodes()
        edge_index = []

        # Iterate over rows to construct edges
        for _, row in self.train_df.iterrows():
            transaction_index = row["node_index"]
            # Create edges between transaction node and all other identity nodes
            for col in node_dict.keys():
                if col != "Transaction" and pd.notna(row[col]):
                    edge_index.append([transaction_index, node_dict[col][row[col]]])
                    edge_index.append(
                        [node_dict[col][row[col]], transaction_index]
                    )  # Adding reverse edge as well

        # Convert to tensor for PyTorch Geometric
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index_tensor

    def pyg_dataset(self):
        # Create PyG dataset object
        print("Creating object for pyg")
        dataset = Data(x=self.features, edge_index=self.edge_index, y=self.labels)
        return dataset
