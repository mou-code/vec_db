from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle



DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
class CustomIndex: pass

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
    # np.memmap() Parameters:
    # First argument: File path to store data
    # dtype: Data type (float32 here)
    # mode: File access mode
    # 'w+': Read/write, create if not exists
    # shape: Dimensions of the array
    # returns file contents on disk
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
            # [0] is necessary because:
            # mmap_vector is 2D array: [[x1, x2, ..., x70]]
            # [0] extracts the single row as 1D: [x1, x2, ..., x70]
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
        index = pickle.load(file)
        file.close()

        cluster_centers=index.centroids
        labels_list=index.labels_list
        for i,vec in enumerate(cluster_centers):
            score=self._cal_score(query,vec)
            scores.append((score,i))
        # # here we assume that the row number is the ID of each vector
        # for row_num in range(num_records):
        #     vector = self.get_one_row(row_num)
        #     score = self._cal_score(query, vector)
        #     scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        # Getting the nearest clusters
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

    def _build_index(self, num_clusters=100, num_subspaces=7):
            
            # Placeholder for index building logic
            ### IVF
            # Apply k means clustering to all vectors -> return the cluster centroids 
            # store the clusters and their centroids in a array or dictionary?
            # Within each cluster:
            # loop over each vector 
            # create an array of subspace arrays (parameter) subspaces[[   [subvector1 of array 1],[subvector1 of array 2]   ] , [],...]
            # apply k-means clustering for each subspace[0] subspace[1] etc..
            # Generate the codebook
            
            
            # Build the PQ-IVF index.
            # Args:
            #     num_clusters (int): Number of clusters for the inverted file (IVF).
            #     num_subspaces (int): Number of subspaces for product quantization.
            #     subspace_dim (int): Dimension of each subspace.
            # Sample dataset
            vectors = self.get_all_rows()

            # Step 1: Coarse Quantization (Clustering)
            kmeans = KMeans(n_clusters=10)
            labels = kmeans.fit_predict(vectors)  # Assign each vector to a cluster
            cluster_centers = kmeans.cluster_centers_
            # Step 2: Construct Posting Lists
            labels_list = {i: [] for i in range(10)}  # Two clusters: 0 and 1
            for i, label in enumerate(labels):
                labels_list[label].append(i)
            
            CustomIndex.centroids=cluster_centers
            CustomIndex.labels_list=labels_list
            
            filehandler = open(self.index_path,"wb")
            pickle.dump(CustomIndex,filehandler)
            filehandler.close()

            # print(object_file.centroids, object_file.labels_list, sep=', ')
            # Print Clustering Results
            # print("Cluster Centers:", cluster_centers)
            # print("Labels:", labels)
            # print("Posting Lists:", labels_list)

            # # # Query Processing
            # query = np.random.randint(0, 10, (1, 7))  # Integers between 0 and 9
            # nearest_centroid = np.argmin([np.linalg.norm(query - centroid) for centroid in kmeans.cluster_centers_])

            # # Fine Search (Within the posting list of nearest centroid)
            # nearest_vectors = [vectors[i] for i in labels_list[nearest_centroid]]
            # closest_vector = min(nearest_vectors, key=lambda x: np.linalg.norm(query - x))
            # print(f"Actual vector: {query}")
            # print(f"Nearest vector: {closest_vector}")
    