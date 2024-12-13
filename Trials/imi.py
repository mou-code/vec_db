from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
import math

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
class IMI:
    @staticmethod
    def build_index(self):
            vectors = self.get_all_rows()

            # Step 1: Coarse Quantization (Clustering)
            n_clusters = self._configure_clusters()          
            part1 = vectors[:,:35]
            part2 = vectors[:,35:]
            # print(part1)
            # print(part2)
            
            # print(vectors)
            # Step 1: Coarse Quantization (Clustering)
            kmeansp1 = KMeans(n_clusters)
            kmeansp1.fit(part1)
            cluster_centersp1 = kmeansp1.cluster_centers_

            kmeansp2 = KMeans(n_clusters)
            kmeansp2.fit(part2)
            cluster_centersp2 = kmeansp2.cluster_centers_

            labelsp1 = kmeansp1.predict(part1)  # Assign each vector's part1 to a cluster
            labelsp2 = kmeansp2.predict(part2)  # Assign each vector's part2 to a cluster     
            
            labels_list = {(i, j): [] for i in range(n_clusters) for j in range(n_clusters)}

            # Assign each vector to the corresponding posting list
            for idx, (label1, label2) in enumerate(zip(labelsp1, labelsp2)):
                labels_list[(label1, label2)].append(idx)
    
            # Save index data as a dictionary
            index_data = {
                "centroidsp1": cluster_centersp1,
                "centroidsp2": cluster_centersp2,
                "labels_list": labels_list,
            }

            with open(self.index_path, "wb") as file:
                pickle.dump(index_data, file)
    @staticmethod
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        query = query.ravel()  # Flattens the query to 1D
        query_part1 = query[:,:35]
        query_part2 = query[:,35:]
        k=10
        file = open(self.index_path,'rb')
        x = pickle.load(file)
        file.close()
        
        cluster_centersp1 = x["cluster_centersp1"]
        cluster_centersp2 = x["cluster_centersp2"]
        labels_list = x["labels_list"]
        # Compute distances to all u's (centroids of part1)
        #calculate distance
        # distances_u = kmeansp1.predict(query_part1)
        distances_u = [np.linalg.norm(query_part1 - centroid) for centroid in cluster_centersp1]

        nearest_u_indices = np.argsort(distances_u)[:k]  # Indices of top-k closest u's
        # Compute distances to all v's (centroids of part2)
        # distances_v = kmeansp2.predict(query_part2)
        distances_v = [np.linalg.norm(query_part2 - centroid) for centroid in cluster_centersp2]

        nearest_v_indices = np.argsort(distances_v)[:k]  # Indices of top-k closest v's
        print ("Nearest U Indices = ",nearest_u_indices," Nearest V indices = ", nearest_v_indices)

        # Combine u's and v's and find the pair with the smallest combined distance
        min_distance = float('inf')
        best_pairs = []
        for u_idx in nearest_u_indices:
            for v_idx in nearest_v_indices:
                combined_distance = distances_u[u_idx] + distances_v[v_idx]
                # if combined_distance < min_distance:
                best_pairs.append((combined_distance,(u_idx,v_idx)))
                # best_pairs.append((u_idx,v_idx))
        best_pairs = sorted(best_pairs, reverse=True)
        taken_values=0
        resulted_vectors=[]
        i=0
        while(taken_values<top_k):
            labels=labels_list[best_pairs[i][1]]
            resulted_vectors.extend(labels_list[best_pairs[i][1]])
            taken_values+=len(labels)
            i+=1
            
        final_vectors=[]
        
        for row_num in resulted_vectors:
                vector = self.get_one_row(row_num)
                vector = vector.ravel()  # Flattens the vector to 1D
                score = self._cal_score(query, vector)
                final_vectors.append((score,row_num))
        # Sort by scores and keep only top_k results
        final_vectors = sorted(final_vectors, reverse=True)[:top_k]
        # print(resulted_vectors)
        # Extract only the row_num from resulted_vectors
        row_nums = [row_num for _, row_num in final_vectors]

        return row_nums  # Return only row_num values    