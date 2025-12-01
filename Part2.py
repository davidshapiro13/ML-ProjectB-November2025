
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection.search import GridSearchCV
from train_valid_test_loader import load_train_valid_test_datasets
import numpy as np
import pandas as pd
import os

train_tuple, valid_tuple, test_tuple, n_users, n_items = \
    load_train_valid_test_datasets()
DATA_PATH = 'data_movie_lens_100k/'
reader1 = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)
reader2 = Reader(rating_scale=(1, 5))
df = pd.read_csv(DATA_PATH + 'ratings_masked_leaderboard_set.csv')
dev_set = Dataset.load_from_file(
  os.path.join(DATA_PATH, 'ratings_all_development_set.csv'), reader=reader1)
training = dev_set.build_full_trainset()


test_set = Dataset.load_from_df(df, reader=reader2)
test = test_set.build_full_trainset().build_testset()

svd = SVD(n_factors=2, n_epochs=15, init_mean=0, lr_all=0.005, reg_all=0.02, random_state=123453)
parameter_grid = dict()
parameter_grid['n_factors'] = {2, 4, 8, 16}
parameter_grid['lr_all'] = np.logspace(-4, 4, 9)
parameter_grid['reg_all'] = np.logspace(-4, 4, 9)

#grid_searcher = GridSearchCV(
#    svd,
#    parameter_grid,
#    measure=['mae'],
    #cv=splitter,
#    refit=True
#)

svd.fit(training)
print(svd.trainset.global_mean)
#print(svd.test(test))
#print(grid_searcher.best_index)