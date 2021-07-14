# This program demos how to connect to Milvus vector database, 
# create a vector collection,
# insert 10 vectors, 
# and execute a vector similarity search.

import random

from milvus import Milvus, IndexType, MetricType, Status

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value
# _PORT = '19121'  # default http value

# Vector parameters
_DIM = 5  # dimension of vector

_INDEX_FILE_SIZE = 1024  # max file size of stored index


def main():
    # Specify server addr when create milvus client instance
    # milvus client instance maintain a connection pool, param
    # `pool_size` specify the max connection num.
    milvus = Milvus(_HOST, _PORT)

    # Create collection demo_collection if it dosen't exist.
    collection_name = 'gomedemo5'

    status, ok = milvus.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.L2  # optional
        }

    # Show collections in Milvus server
    _, collections = milvus.list_collections()

    # Describe demo_collection
    _, collection = milvus.get_collection_info(collection_name)
    print(collection)

    # Get demo_collection row count
    status, result = milvus.count_entities(collection_name)

    # present collection statistics info
    _, info = milvus.get_collection_stats(collection_name)
    print(info)

    # describe index, get information of index
    status, index = milvus.get_index_info(collection_name)
    print(index)

    # Use the top 10 vectors for similarity search

    # execute vector similarity searchpip install --upgrade pi
    search_param = {
        "nprobe": 16
    }

    print("Searching ... ")
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10)]
    # query_vectors = vectors[0:10]
    query_vectors = [[2, 2, 3, 2, 10]]
    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,
        'top_k': 3,
        'params': search_param,
    }

    status, results = milvus.search(**param)

    print("Searching 2... ")

    if status.OK():
        # indicate search result
        # also use by:
        #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
        # print results
        print(results)
    else:
        print("Search failed. ", status)


if __name__ == '__main__':
    main()
# ids = ["1624953490933507009","1624953490933507014"]
