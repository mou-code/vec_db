# Semantic Search Engine with Vectorized Databases
This repository is for a semantic search engine that utilizes vectorized databases to retrieve information based on vector space embeddings. The project is designed to build an efficient indexing system, with a focus on optimizing query response time, memory usage, and index size. It aims to balance the limitations of RAM, query score accuracy, and the overall time required for retrieval, ensuring a performant and scalable solution for semantic search. 

## Project Overview

The key components of the project include:
- `VecDB`: A class representing the vectorized database, responsible for storing and retrieving vectors.
- `generate_database()`: A method to generate a random database.
- `get_one_row()`: A method to get one row from the database given its index.
- `insert_records()`: A method to insert multiple records into the database. It then rebuilds the index.
- `retrieve()`: A method to retrieve the top-k most similar based on a given query vector.
- `_cal_score()`: A helper method to calculate the cosine similarity between two vectors.
- `_build_index()`: A placeholder method for implementing an indexing mechanism.

## Implemented Approach
A multi-level IVF (Inverted File) structure enhances indexing efficiency through hierarchical clustering. Initially, the vector space is partitioned using standard IVF techniques at the first level. Each resulting cluster is then subdivided at the second level using MiniBatchKMeans, where the number of second-level clusters is determined by the square root of the number of vectors in each cluster. To handle varying dataset sizes, the index structure is adapted based on different seed data, with the number of clusters at the first level changing according to the dataset size.

This approach strikes a balance between reduced memory usage and improved retrieval accuracy, all while ensuring scalable query performance. However, reading entire cluster vectors at once, while speeding up the process, can lead to high memory consumption. To address this trade-off between time and memory, we adopted a strategy of reading cluster vectors in batches. This approach ensures more efficient memory management while still maintaining a reasonable processing time for queries.

## Usage

The project provides a `VecDB` class that you can use to interact with the vectorized database. Here's an example of how to use it:

```python
import numpy as np
from vec_db import VecDB

# Create an instance of VecDB and random DB of size 10K
db = VecDB(db_size = 10**4)

# Retrieve similar images for a given query
query_vector = np.random.rand(1,70) # Query vector of dimension 70
similar_images = db.retrieve(query_vector, top_k=5)
print(similar_images)
```
## Samples with different query seeds
<table>
  <tr>
    <td align="center">
      Query seed=10
    </td>
    <td>
    <img src="https://github.com/user-attachments/assets/b84d42c4-6731-429b-abf8-1d3125e3bfcd" width="500px;">
    </td>
  </tr>
  <tr>
    <td align="center">
      Query seed=39
    </td>
    <td>
    <img src="https://github.com/user-attachments/assets/75eda7df-5cbb-4b83-99f7-3ec762b92245" width="500px;">
    </td>
  </tr>
    <tr>
    <td align="center">
      Query seed=100
    </td>
    <td>
    <img src="https://github.com/user-attachments/assets/92e844b1-c6cd-45c2-9927-ac89a89f0370" width="500px;">
    </td>
  </tr>
    <tr>
    <td align="center">
      Query seed=285616
    </td>
    <td>
    <img src="https://github.com/user-attachments/assets/feb32739-f39e-4347-80ed-dcc47e9660ed" width="500px;">
    </td>
  </tr>
 </table>

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Menna-Ahmed7" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110634473?v=4" width="150px;" alt="https://github.com/Menna-Ahmed7"/>
    <br />
    <sub><b>Mennatallah Ahmed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MostafaBinHani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119853216?v=4" width="150px;" alt="https://github.com/MostafaBinHani"/>
    <br />
    <sub><b>Mostafa Hani</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohammadAlomar8" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119791309?v=4" width="150px;" alt="https://github.com/MohammadAlomar8"/>
    <br />
    <sub><b>Mohammed Alomar</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mou-code" target="_black">
    <img src="https://avatars.githubusercontent.com/u/123744354?v=4" width="150px;" alt="https://github.com/mou-code"/>
    <br />
    <sub><b>Moustafa Mohammed</b></sub></a>
    </td>
  </tr>
 </table>
