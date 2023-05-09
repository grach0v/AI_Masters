import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
import shap

def plot_scores(X_tr, y_prob_tr, X_val, y_prob):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))

    ax[0].set_title('train scores', fontsize=15)
    ax[0].set_xlabel('model score', fontsize=15)
    ax[0].set_ylabel('Count', fontsize=15)
    ax[0].set_yscale('log')
    ax[0].tick_params('both', labelsize=15)
    sns.histplot(X_tr, x=y_prob_tr, hue='is_callcenter', bins=25, ax=ax[0])

    ax[1].set_title('val scores', fontsize=15)
    ax[1].set_xlabel('model score', fontsize=15)
    ax[1].set_ylabel('Count', fontsize=15)
    ax[1].set_yscale('log')
    ax[1].tick_params('both', labelsize=15)
    _ = sns.histplot(X_val, x=y_prob, hue='is_callcenter', bins=25, ax=ax[1])
    
    plt.show()

    
def plot_curves(y_train, y_prob_train, y_val, y_prob_val):
    fig, ax = plt.subplot_mosaic([['pr', 'roc']], figsize=(16, 5))
    
    ax['pr'].set_title('PR curve', fontsize=15)
    prec, rec, thrs = precision_recall_curve(y_train, y_prob_train)
    ax['pr'].plot(rec, prec, lw=3, label='train')
    ax['pr'].plot(rec[1:], thrs, lw=3, label='threshold on train', ls='--', color='red')
    
    prec, rec, thrs = precision_recall_curve(y_val, y_prob_val)
    ax['pr'].plot(rec, prec, lw=3, label='val')
    
    
    ax['roc'].set_title('ROC curve', fontsize=15)
    fpr, tpr, thrs  = roc_curve(y_train, y_prob_train)
    ax['roc'].plot(fpr, tpr, lw=3, label='train')
    ax['roc'].plot(fpr[1:], thrs[1:], lw=3, label='threshold on train', ls='--', color='red')
    
    fpr, tpr, thrs  = roc_curve(y_val, y_prob_val)
    ax['roc'].plot(fpr, tpr, lw=3, label='val')
    
    ax['pr'].set_xlabel('Recall', fontsize=15)
    ax['pr'].set_ylabel('Precision', fontsize=15)
    ax['pr'].tick_params('both', labelsize=15)
    ax['pr'].legend(fontsize=15)
    
    ax['roc'].set_xlabel('FPR', fontsize=15)
    ax['roc'].set_ylabel('TPR', fontsize=15)
    ax['roc'].tick_params('both', labelsize=15)
    ax['roc'].legend(fontsize=15)
    
    plt.show()

    
def get_shap(model, X, target, features, max_display=30):
    exp = shap.Explainer(model)
    dummy = X[np.r_[[target], features]].copy()

    for col in dummy.columns[dummy.dtypes == 'category']:
        dummy[col] = dummy[col].cat.codes

    shap_values = exp(dummy.groupby(target).apply(lambda x: x.sample(min(x.shape[0], 300)))[features])
    shap.summary_plot(shap_values[:, :, 1], max_display=max_display, color_bar=False)
    