import lightgbm as lgb
import numpy as np
from preprocess import preprocess_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
# import seaborn as sns  # data visualization lib
import matplotlib.pyplot as plt


def train_model():

    x_train, x_test, y_train, y_test = preprocess_data()
    clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=18, max_leaf_nodes=64, verbose=1,
                                 n_jobs=4)
    scores_rfc = []
    # models1 = []
    # initialize KFold, we vcan use stratified KFold to keep the same imblance ratio for target
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    for i, (train_idx, valid_idx) in enumerate(kf.split(x_train, y_train)):
        print('...... training {}th fold \n'.format(i + 1))
        tr_x = x_train[train_idx]
        tr_y = y_train[train_idx]

        val_x = x_train[valid_idx]
        val_y = y_train[valid_idx]
        model = clf
        model.fit(tr_x, tr_y)
        # picking best model?
        pred_val_y = model.predict(val_x)
        # measuring model vs validation
        score_rfc = roc_auc_score(val_y, pred_val_y)
        scores_rfc.append(score_rfc)
        print('current performance by auc:{}'.format(score_rfc))

        # auc_scores1.append(auc)
        # models1.append(model)
    best_f1 = -np.inf
    best_thred = 0
    v = [i * 0.01 for i in range(50)]
    for thred in v:
        preds = (pred_val_y > thred).astype(int)
        f1 = f1_score(val_y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thred = thred

    y_pred_rfc = (pred_val_y > best_thred).astype(int)
    print(confusion_matrix(val_y, y_pred_rfc))
    print(f1_score(val_y, y_pred_rfc))
    print('the average mean auc is:{}'.format(np.mean(scores_rfc)))
    model_lgb = lgb.LGBMClassifier(n_jobs=4, n_estimators=10000, boost_from_average='false', learning_rate=0.01,
                                   num_leaves=64, num_threads=4, max_depth=-1, tree_learner="serial",
                                   feature_fraction=0.7, bagging_freq=5, bagging_fraction=0.7, min_data_in_leaf=100,
                                   silent=-1, verbose=-1, max_bin=255, bagging_seed=11, )
    auc_scores = []
    models = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
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
    print('the average mean auc is:{}'.format(np.mean(auc_scores)))
    fpr, tpr, _ = roc_curve(va_y, pred_va_y)
    # plot model roc curve
    plt.plot(fpr, tpr, marker='.', label='LGB model')
    # axis labels
    plt.title('ROC AUC CURVE')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig('LGB ROC_auc_curve.png')
    plt.show()
    # Test data
    pred_test_1 = models[0].predict_proba(x_test, num_iteration=models[0].best_iteration_)[:, 1]
    pred_test_2 = models[1].predict_proba(x_test, num_iteration=models[1].best_iteration_)[:, 1]
    pred_test_3 = models[2].predict_proba(x_test, num_iteration=models[2].best_iteration_)[:, 1]
    pred_test_4 = models[3].predict_proba(x_test, num_iteration=models[3].best_iteration_)[:, 1]
    pred_test_5 = models[4].predict_proba(x_test, num_iteration=models[4].best_iteration_)[:, 1]
    pred_test = (pred_test_1 + pred_test_2 + pred_test_3 + pred_test_4 + pred_test_5) / 5.0
    print(pred_test)


if __name__ == '__main__':
    train_model()
