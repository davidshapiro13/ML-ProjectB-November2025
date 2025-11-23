
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVD
from sklearn.model_selection import GridSearchCV
from train_valid_test_loader import load_train_valid_test_datasets
import numpy as np
import os

DATA_PATH = 'data_movie_lens_100k/'
reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)
dev_set = Dataset.load_from_file(
  os.path.join(DATA_PATH, 'ratings_all_development_set.csv'), reader=reader)
test_set = Dataset.load_from_file(
  os.path.join(DATA_PATH, 'ratings_masked_leaderboard_set.csv'), reader=reader)

svd = SVD(n_factors=2, n_epochs=15, init_mean=0, lr_all=0.005, reg_all=0.02, random_state=123453)
parameter_grid = dict()
parameter_grid['n_factors'] = {2, 4, 8, 16}
parameter_grid['lr_all'] = np.logspace(-4, 4, 9)
parameter_grid['reg_all'] = np.logspace(-4, 4, 9)

grid_searcher = GridSearchCV(
    svd,
    parameter_grid,
    scoring='MAE',
    #cv=splitter,
    refit=True
)
svd.fit(dev_set)
yhat = svd.test(test_set)
print(yhat)