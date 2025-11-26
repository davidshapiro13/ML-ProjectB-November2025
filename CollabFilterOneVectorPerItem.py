'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets
from sklearn.model_selection import GridSearchCV

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object
        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.asarray([ag_np.mean(train_tuple[2])]),
            b_per_user=ag_np.ones(n_users), # FIX dimensionality
            c_per_item=ag_np.ones(n_items), # FIX dimensionality
            U=0.001 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.001 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        N = user_id_N.size
        yhat_N = ag_np.zeros(0)
        for i, j in zip(user_id_N, item_id_N):
            new_vals = mu + b_per_user[i] + c_per_item[j] + ag_np.dot(U[i, :].reshape(self.n_factors, 1).T, V[j, :].reshape(self.n_factors, 1))
            yhat_N = ag_np.hstack([yhat_N, new_vals[0]])
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        loss_total = self.alpha * (ag_np.sum(ag_np.square(param_dict['V'])) + ag_np.sum(ag_np.square(param_dict['U']))) + ag_np.sum(ag_np.square(y_N - yhat_N))
        return loss_total    

rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
scoring = {'RMSE' : rmse_scorer, 'MAE' : mae_scorer}
if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=1000, step_size=0.1,
        n_factors=2, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    find_best_step(n_users, n_items, train_tuple, valid_tuple)

    RMSE_train_per_K = []
    RMSE_valid_per_K = []
    MAE_train_per_K = []
    MAE_valid_per_K = []
    models = []
    #Alpha = 0
    '''
    for K in [2, 10, 50]:
        model = CollabFilterOneVectorPerItem(
        n_epochs=20, batch_size=1000, step_size=0.01,
        n_factors=K, alpha=0.0)
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)
        train_1 = model.trace_rmse_train
        valid_1 = model.trace_rmse_valid
        #plt.plot( model.trace_epoch, model.trace_rmse_train, '.-', color='blue', label='Training')
        #plt.plot( model.trace_epoch, model.trace_rmse_valid,  '.-', color='red', label="Validation")
        #plt.ylabel('RMSE')
        #plt.xlabel("Epoch")
        #plt.title("K = " + str(K))
        #plt.legend(bbox_to_anchor=(1.33, 0.5)) # make legend outside plot
        #plt.show()
        RMSE_train_per_K.append(model.trace_rmse_train[-1])
        RMSE_valid_per_K.append(model.trace_rmse_valid[-1])
        MAE_train_per_K.append(model.trace_mae_train[-1])
        MAE_valid_per_K.append(model.trace_mae_valid[-1])
        models.append(model)
    best_K_RMSE_index = RMSE_valid_per_K.index(min(RMSE_valid_per_K))
    best_K_MAE_index = MAE_valid_per_K.index(min(MAE_valid_per_K))
    print("Best K index for RMSE", best_K_RMSE_index)
    print("Best K index for MAE", best_K_MAE_index)
    #Best Model is index 2, K=5

for K in [2, 10, 50]:
    model = CollabFilterOneVectorPerItem(
        n_epochs=400, batch_size=1000, step_size=0.1,
        n_factors=K, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)
    parameter_grid = dict()
    parameter_grid['n_factors'] = {2, 10, 50}
    parameter_grid['step_size'] = {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1}

    grid_searcher = GridSearchCV(
        model,
        parameter_grid,
        scoring=scoring,
        #cv=splitter,
        refit=True
    )
    grid_searcher.fit(train_tuple, valid_tuple)

grid_searcher.best_params_

    best_model = CollabFilterOneVectorPerItem(
        n_epochs=400, batch_size=1000, step_size=0.1,
        n_factors=50, alpha=0.0)
    best_model.init_parameter_dict(n_users, n_items, train_tuple)
    best_model.fit(train_tuple, valid_tuple)
    training_metrics = best_model.evaluate_perf_metrics(*train_tuple)
    validation_metrics = best_model.evaluate_perf_metrics(*valid_tuple)
    test_metrics = best_model.evaluate_perf_metrics(*test_tuple)

    MAE_train = training_metrics['mae']
    MAE_valid = validation_metrics['mae']
    MAE_test = test_metrics['mae']

    RMSE_train = training_metrics['rmse']
    RMSE_valid = validation_metrics['rmse']
    RMSE_test  = test_metrics['rmse']

    print("MAE")
    print("training: ", MAE_train)
    print("validation: ", MAE_valid)
    print("test: ", MAE_test)

    print("RMSE")
    print("training: ", RMSE_train)
    print("validation: ", RMSE_valid)
    print("test: ", RMSE_test)

    #Alpha varying

    model2 = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=1000, step_size=0.1,
        n_factors=50, alpha=0.0)
    model2.init_parameter_dict(n_users, n_items, train_tuple)

    parameter_grid['alpha'] = ag_np.logspace(-4, 4, 9)
    grid_searcher2 = GridSearchCV(
        model2,
        parameter_grid,
        scoring=scoring,
        #cv=splitter,
        refit='RMSE'
    )
    grid_searcher2.fit(train_tuple, valid_tuple)
    print("Best Params 2: ", grid_searcher2.best_params_)
    print("Best Score 2: ", grid_searcher2.best_score_)
'''