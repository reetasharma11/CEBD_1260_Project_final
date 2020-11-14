import lightgbm as lgb
import numpy as np
import pickle
from preprocess import preprocess_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score,confusion_matrix


def train_model():
    x_train, x_test, y_train, y_test = preprocess_data()

    # building model
    model_lgb = lgb.LGBMClassifier(
        n_jobs=4,
        n_estimators=10000,
        boost_from_average='false',
        learning_rate=0.01,
        num_leaves=64,
        num_threads=4,
        max_depth=-1,
        tree_learner="serial",
        feature_fraction=0.7,
        bagging_freq=5,
        bagging_fraction=0.7,
        min_data_in_leaf=100,
        silent=-1,
        verbose=-1,
        max_bin=255,
        bagging_seed=11,
    )
    # initialize KFold, we can use stratified KFold to keep the same imbalance ratio for target
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)
    auc_scores = []  # save auc score for each fold
    models = []  # save model for each fold
    for i, (train_idx, valid_idx) in enumerate(kf.split(x_train, y_train)):
        print('...... training {}th fold \n'.format(i + 1))
        tr_x = x_train[train_idx]
        tr_y = y_train[train_idx]

        va_x = x_train[valid_idx]
        va_y = y_train[valid_idx]

        model = model_lgb  # you need to initialize your lgb model at each loop, otherwise it will overwrite
        model.fit(tr_x, tr_y, eval_set=[(tr_x, tr_y), (va_x, va_y)], eval_metric='auc', verbose=500,
                  early_stopping_rounds=300)

        # calculate current auc after training the model
        pred_va_y = model.predict_proba(va_x, num_iteration=model.best_iteration_)[:, 1]
        auc = roc_auc_score(va_y, pred_va_y)
        print('current best auc score is:{}'.format(auc))
        auc_scores.append(auc)
        models.append(model)

        best_f1 = -np.inf
        best_thred = 0
        v = [i * 0.01 for i in range(50)]
        for thred in v:
            preds = (pred_va_y > thred).astype(int)
            f1 = f1_score(va_y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thred = thred

        y_pred_lgb = (pred_va_y > best_thred).astype(int)
        print(confusion_matrix(va_y, y_pred_lgb))
        print(f1_score(va_y, y_pred_lgb))
        with open('model.pkl', 'wb') as handle:
            pickle.dump(model_lgb, handle)
        handle.close()
    print('the average mean auc is:{}'.format(np.mean(auc_scores)))

if __name__ == '__main__':
    train_model()
