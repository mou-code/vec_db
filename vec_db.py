from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle

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

        n_probe = 5
        cluster_scores = sorted(scores, reverse=True)[:n_probe]
        # Get the vectors of nearest clusters
        top_vector=[]
        # for item in scores:
        top_vector.append(labels_list[cluster_scores[0][1]])
        # print("top_vector",top_vector[0])
        
        # store the vectors of the top cluster
        resulted_vectors=[]
        #loop over top_vectors and cosine similarity
        for row_num in top_vector[0]:
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
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def configure_clusters(num_records):
        match num_records:
            case 1_000_000:
                n_clusters = 256
            case 10_000_000:
                n_clusters = 2560
            case 15_000_000:
                n_clusters = 3750
            case 20_000_000:
                n_clusters = 5000
        print(f"Number of clusters: {n_clusters}")
        return n_clusters  
             
    def _build_index(self):
            vectors = self.get_all_rows()

            # Step 1: Coarse Quantization (Clustering)
            num_records = self._get_num_records()
            n_clusters = self.configure_clusters(num_records)               
             
            kmeans = KMeans(n_clusters)
            labels = kmeans.fit_predict(vectors)  # Assign each vector to a cluster
            cluster_centers = kmeans.cluster_centers_
            
            # Step 2: Construct Posting Lists
            labels_list = {i: [] for i in range(n_clusters)}  # Two clusters: 0 and 1
            for i, label in enumerate(labels):
                labels_list[label].append(i)
            
            # Save index data as a dictionary
            index_data = {
                "centroids": cluster_centers,
                "labels_list": labels_list,
            }

            with open(self.index_path, "wb") as file:
                pickle.dump(index_data, file)


