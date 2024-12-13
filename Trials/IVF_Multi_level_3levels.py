from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle
import os
import math
import shutil
import joblib
import gzip
def build_index_level_1_clustering(vectors,n_clusters,index_path):
    chunk_size=10**6
    firstlevel_nclusters = n_clusters
    if len(vectors)==10**6:
          batch_size=10
    else: batch_size=10**4
    kmeans = MiniBatchKMeans(
                n_init=10,
                n_clusters=firstlevel_nclusters,
                batch_size=batch_size,  # Smaller batch size
                max_iter=700  # Fewer iterations
                )
    kmeans.fit(vectors[:10**6])
    # cluster_centers_level1 = kmeans.cluster_centers_

    # print(cluster_centers_level1)
    labels = []
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i + chunk_size]
        labels.extend(kmeans.predict(chunk))
    # print("labels=",labels)
    # labels = kmeans.predict(vectors)  # Assign each vector to a cluster
    # Step 2: Construct Posting Lists
    labels_array = np.array(labels) # Two clusters: 0 and 1
    # Calculate the cluster sizes
    cluster_sizes = np.bincount(labels_array, minlength=firstlevel_nclusters)
    # Filter out clusters with no data
    valid_clusters_mask = cluster_sizes > 0
    filtered_centroids = kmeans.cluster_centers_[valid_clusters_mask]
    # Use a dictionary comprehension with NumPy masking to create the filtered labels_list
    filtered_labels_list = {
        cluster: np.where(labels_array == cluster)[0].tolist()
        for cluster in np.nonzero(valid_clusters_mask)[0]
    }

    index_path_level1=f"level1_centroids_{index_path}"
    index_data = {
                    "centroids": filtered_centroids,
                }
    with open(index_path_level1, "wb") as file:
                    pickle.dump(index_data, file)
    # print(labels_list)
    return filtered_labels_list
def build_index_level_2_clustering(vectors,labels_list,index_path):
    folder_path = f"level2_centroids_{index_path}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if len(vectors)==10**6:
          batch_size=10
    else: batch_size=10**4
    kmeans_dict = {}
    # Create a MiniBatchKMeans instance for each cluster index
    for cluster_idx in labels_list:
        # number of clusters in level 2 within each cluster of level1=root(#vectors in cluster)
        length_vectors=len(labels_list[cluster_idx])
        kmeans_dict[cluster_idx] = MiniBatchKMeans(
            n_clusters=int(math.sqrt(length_vectors)),
            batch_size=batch_size,
            max_iter=100
        )
    print("labels_list",labels_list)

    for i in labels_list:
        # print("cluster=",i)
        length_vectors=len(labels_list[cluster_idx])
        secondlevel_nclusters=int(math.sqrt(length_vectors))
        vectors_of_cluster=labels_list[i]

        kmeans_dict[i].fit(vectors[vectors_of_cluster])
        cluster_centers_level2 = kmeans_dict[i].cluster_centers_
        # print("cluster_centers_level2",cluster_centers_level2)
        # Predict labels for the current cluster's vectors
        labels_l2=kmeans_dict[i].predict(vectors[vectors_of_cluster])
        labels_array_l2 = np.array(labels_l2)
        # print("labels_array_l2",labels_array_l2)
        # Calculate cluster sizes and filter empty clusters
        cluster_sizes_l2 = np.bincount(labels_array_l2, minlength=secondlevel_nclusters)
        valid_clusters_mask_l2 = cluster_sizes_l2 > 0
        valid_indices = np.nonzero(valid_clusters_mask_l2)[0]
        # print("valid_indices",valid_indices)
        # Filter centroids for level 2 clusters
        filtered_centroids_l2 = cluster_centers_level2[valid_indices]

        # Create filtered labels_list for valid clusters
        filtered_labels_list_level_2 = {
            cluster: [vectors_of_cluster[idx] for idx in np.where(labels_array_l2 == cluster)[0]]
            for cluster in valid_indices
            }
        # Create the file path inside the folder
        index_data = {
                        "centroids": filtered_centroids_l2.astype(np.float16),
                    }
        file_path=os.path.join(folder_path,f"cluster_{i}")
        print(file_path)
        with gzip.open(file_path, "wb") as file:
                joblib.dump(index_data, file,compress=9)
    return filtered_labels_list_level_2

def build_index_level_3_clustering(vectors,labels_list,index_path): 
    folder_path = f"third_level_centroids_{index_path}"

    # Remove the folder if it exists
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # Create the folder
    os.makedirs(folder_path)
    if len(vectors)==10**6:
          batch_size=10
    else: batch_size=10**4
    kmeans_dict = {}
    # Create a MiniBatchKMeans instance for each cluster index
    for cluster_idx in labels_list:
        # number of clusters in level 2 within each cluster of level1=root(#vectors in cluster)
        length_vectors=len(labels_list[cluster_idx])
        kmeans_dict[cluster_idx] = MiniBatchKMeans(
            n_init=10,
            batch_size=batch_size,  # Smaller batch size
            max_iter=500,
            n_clusters=int(math.sqrt(length_vectors)),

        )
    # print("labels_list",labels_list)
    # chunk_size_level_2=10
    for i in labels_list:
        # print("cluster=",i)
        length_vectors=len(labels_list[cluster_idx])
        thirdlevel_nclusters=int(math.sqrt(length_vectors))
        vectors_of_cluster=labels_list[i]

        kmeans_dict[i].fit(vectors[vectors_of_cluster])
        cluster_centers_level3 = kmeans_dict[i].cluster_centers_
        # print("cluster_centers_level2",cluster_centers_level2)
        # Predict labels for the current cluster's vectors
        labels_l3=kmeans_dict[i].predict(vectors[vectors_of_cluster])
        labels_array_l3 = np.array(labels_l3)
        # print("labels_array_l2",labels_array_l2)
        # Calculate cluster sizes and filter empty clusters
        cluster_sizes_l3 = np.bincount(labels_array_l3, minlength=thirdlevel_nclusters)
        valid_clusters_mask_l3 = cluster_sizes_l3 > 0
        valid_indices = np.nonzero(valid_clusters_mask_l3)[0]
        # print("valid_indices",valid_indices)
        # Filter centroids for level 2 clusters
        filtered_centroids_l2 = cluster_centers_level3[valid_indices]

        # Create filtered labels_list for valid clusters
        filtered_labels_list_level_2 = {
            cluster: [vectors_of_cluster[idx] for idx in np.where(labels_array_l3 == cluster)[0]]
            for cluster in valid_indices
            }
        # Create the file path inside the folder
        index_path_level2 = os.path.join(folder_path, f"level3_centroids_cluster{i}")
        index_data = {
                        "centroids": filtered_centroids_l2.astype(np.float16),
                        "labels_list":filtered_labels_list_level_2
                    }
        
        with gzip.open(index_path_level2, "wb") as file:
                joblib.dump(index_data, file,compress=9)