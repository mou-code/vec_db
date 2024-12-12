from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import pickle
import math
from heapq import heappush, heappop
from IVF_Multi_level import build_index_level_1_clustering,build_index_level_2_clustering

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def level_2_query(self,file_path,query,n_probes,top_k_heap):
        file = open(file_path,'rb')
        level2_data_loaded = pickle.load(file)
        file.close()

        num_records=self._get_num_records()
        level2_centroids_loaded=level2_data_loaded["centroids"]
        level2_labels_loaded=level2_data_loaded["labels_list"]
        # print("level2_labels_loaded",level2_labels_loaded)
        nearest_centroids_level_2 = sorted([(np.linalg.norm(query - centroid),i) for i,centroid in enumerate(level2_centroids_loaded)])
        n_probe_l2=min(n_probes,len(nearest_centroids_level_2))
        # print("nearest_centroids_level_2",nearest_centroids_level_2)
        nearest_centroids_level_2=nearest_centroids_level_2[:n_probe_l2]
        # print(nearest_centroids_level_2)

        batch_size=100
        for _,centroid_idx_l2 in nearest_centroids_level_2:
           # Check if centroid_idx_l2 exists as a key in level2_labels_loaded
          if centroid_idx_l2 in level2_labels_loaded:
            row_indices = sorted(level2_labels_loaded[centroid_idx_l2]) 
            rows_length=len(row_indices)
            # Get data of cluster in batches
            for start_i in range(0, rows_length, batch_size):
              end_i = min(start_i + batch_size, rows_length)
              subset_indices = row_indices[start_i:end_i]
              # Get first vector number in the batch to be the offset to start from 
              start_offset = subset_indices[0] * DIMENSION * ELEMENT_SIZE
              # To get the shape of the part to point at in data file
              remaining_records = num_records - subset_indices[0]
              mmap_vectors = np.memmap(self.db_path,dtype=np.float32,mode='r',shape=(remaining_records, DIMENSION),
              offset=start_offset)

              # As in returned memmap the indices are numbered 0 starting from start offset
              relative_indices = np.array(subset_indices) - subset_indices[0]
              cluster_vectors = mmap_vectors[relative_indices]
              # print("l",level2_labels_loaded[centroid_idx_l2])
              for vec_num, vector in zip(subset_indices, cluster_vectors):
                  score=self._cal_score(vector,query)
                  heappush(top_k_heap, (-score, vec_num))
        return top_k_heap

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        query = query.ravel()  # Flattens the query to 1D
        num_records = self._get_num_records()
        saved_db_name=self.index_path[:-4]
        folder_path = f"{saved_db_name}/second_level_centroids_{self.index_path}"
        index_path_level1=f"{saved_db_name}/level1_centroids_{self.index_path}"

        if num_records == 10**6:
           n_probe =40
        elif num_records ==10**7:
          n_probe=20
        elif num_records == 15*10**6:
          n_probe=20
        elif num_records==20*10**6:
            n_probe=8
        # print("n_probe=",n_probe)
        # 1. Getting nearest centroids in first level
        #get index data
        file = open(index_path_level1,'rb')
        level1_centroids_loaded = pickle.load(file)["centroids"]
        file.close()
        # print("level1_centroids",level1_centroids_loaded)
        nearest_centroids = sorted([(np.linalg.norm(query - centroid),i) for i,centroid in enumerate(level1_centroids_loaded)])
        # print(nearest_centroids)
        nearest_centroids=nearest_centroids[:n_probe]

        # 2. Getting nearest centroids in second level
        top_k_heap=[]
        if num_records == 10**6:
           n_probes_l2 =80
        elif num_records ==10**7:
          n_probes_l2=120
        elif num_records == 15*10**6:
          n_probes_l2=310
        elif num_records==20*10**6:
            n_probes_l2=30
        for _,centroid_idx in nearest_centroids:
            # print("cluster num#",centroid_idx)
            # now within a cluster, let's open file of the cluster
            file_path=f"{folder_path}/level2_centroids_cluster{centroid_idx}"
            if not os.path.exists(file_path): # no data in this centroid from level1
              continue
            top_k_heap=self.level_2_query(file_path,query,n_probes_l2,top_k_heap)
    
        resulted_vectors = top_k_heap[:top_k]  # Restore original score
        row_nums = [row_num for _, row_num in resulted_vectors]
        return row_nums  # Return only row_num values   
         
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _configure_clusters(self):
        num_records = self._get_num_records()
        match num_records:
            case 1_000_000:
                n_clusters = 256
            case 10_000_000:
                n_clusters = 2600
            case 15_000_000:
                n_clusters = 3750
            case 20_000_000:
                n_clusters = 4400
        print(f"Number of clusters: {n_clusters}")
        return n_clusters
    
    def _build_index(self):
            vectors = self.get_all_rows()
            chunk_size=10**6
            # Step 1: Coarse Quantization (Clustering)
            n_clusters = self._configure_clusters()               
            build_index_level_2_clustering(vectors,build_index_level_1_clustering(vectors,n_clusters,self.index_path),self.index_path)
            
