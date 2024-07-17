import multiprocessing as mp
import time

def test(a,b):
    print(a+b)
    
if __name__ == "__main__":
    num_folds = 10
    search_space_combinations = [(i, j) for i in range(1, 101) for j in range(i, 101)]
    # print(search_space_combinations)
    parallel = 3
    # search_space_chunks = [search_space_combinations[i::parallel] for i in range(parallel)]
    t1 = time.time()
    for i in range(len(search_space_combinations)):
        test(i[0],i[1])
    t2 = t1 - time.time()

    # Using multiprocessing.Pool to parallelize cross-validation
    t3 = time.time()
    with mp.Pool(processes=parallel) as pool:
        # pool.starmap(hyperparameter_tuning, search_space_combinations)
        pool.starmap(test, search_space_combinations)
    t4 = t3 - time.time()

    print(f'seque time:{t2}')
    print(f'paral time:{t4}')