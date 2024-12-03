from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
import math

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class IVF:
    def _build_index(self):
            vectors = self.get_all_rows()
            chunk_size=10**6
            no_chunks=math.ceil(self._get_num_records()/chunk_size)
            # Step 1: Coarse Quantization (Clustering)
            n_clusters = self._configure_clusters()               
            
            kmeans = KMeans(n_clusters)
            kmeans.fit(vectors[:1*10**6])
            cluster_centers = kmeans.cluster_centers_
            # for i in range(no_chunks):
            labels = kmeans.predict(vectors)  # Assign each vector to a cluster
            
            # Step 2: Construct Posting Lists
            labels_list = {i: [] for i in range(n_clusters)}  # Two clusters: 0 and 1
            for i, label in enumerate(labels):
                labels_list[label].append(i)
            print("labels=",labels_list)
            # Save index data as a dictionary
            index_data = {
                "centroids": cluster_centers,
                "labels_list": labels_list,
            }

            with open(self.index_path, "wb") as file:
                pickle.dump(index_data, file)
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        query = query.ravel()  # Flattens the query to 1D
        num_records = self._get_num_records()
        file = open(self.index_path,'rb')
        x = pickle.load(file)
        file.close()
        
        cluster_centers = x["centroids"]
        labels_list = x["labels_list"]
        for i,vec in enumerate(cluster_centers):
            score=self._cal_score(query,vec)
            scores.append((score,i))

        n_probe = 6
        cluster_scores = sorted(scores, reverse=True)[:n_probe]
        # Get the vectors of nearest clusters
        top_vector=[]
        # for item in scores:
        for i in range(n_probe):
            top_vector.append(labels_list[cluster_scores[i][1]])

        # print("top_vector",top_vector[0])
        # store the vectors of the top cluster
        resulted_vectors=[]
        #loop over top_vectors and cosine similarity
        for i in range(n_probe):
            for row_num in top_vector[i]:
                vector = self.get_one_row(row_num)
                vector = vector.ravel()  # Flattens the vector to 1D
                score = self._cal_score(query, vector)
                resulted_vectors.append((score,row_num))
        # Sort by scores and keep only top_k results
        resulted_vectors = sorted(resulted_vectors, reverse=True)[:top_k]
        # print(resulted_vectors)
        # Extract only the row_num from resulted_vectors
        row_nums = [row_num for _, row_num in resulted_vectors]

        return row_nums  # Return only row_num values   