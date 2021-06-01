

X = X_train_corpus_tfidf_df
Y = Y_train

Xv = X_validate_corpus_tfidf_df
Yv = Y_validate

############################################################
# Multi-label random forest
############################################################

colormap = plt.cm.jet(np.linspace(0,1,Y.shape[1]))

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

rfc = RandomForestClassifier(random_state=0, class_weight='balanced', verbose=1)

# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#rfc_grid_param = {'n_estimators': [10,50,100,500],'criterion': ['entropy'],'bootstrap': [True]} # max_features, max_depth, max_leaf_nodes 
rfc_grid_param = {'n_estimators': [100],'criterion': ['entropy'],'bootstrap': [True]} # max_features, max_depth, max_leaf_nodes 
rfc_gd_sr = GridSearchCV(estimator=rfc, param_grid=rfc_grid_param, scoring='roc_auc', cv=3,n_jobs=1, refit =True, return_train_score = True)
rfc_gd_sr.fit(X=X, y=Y)
print(rfc_gd_sr.best_params_)
print(rfc_gd_sr.best_score_)  
print(rfc_gd_sr.cv_results_ )

rfc_gd_sr.cv_results_['mean_test_score']
rfc_gd_sr.cv_results_['rank_test_score']
rfc_gd_sr.cv_results_['std_test_score']

###########
# ROC & AUC 
###########

rfc = RandomForestClassifier(**rfc_gd_sr.best_params_)
#rfc = RandomForestClassifier(random_state=0, class_weight='balanced', verbose=1, n_estimators=100, criterion='entropy', bootstrap=True)
rfc.fit(X=X, y=Y)
Yv_p = rfc.predict_proba(Xv)
category_auc_rfc = []
Yv_category_prob = pd.DataFrame()
i=0
for i in range(Yv.shape[1]):
    print(i)
    Yv_category_prob[Yv.columns[i]] = copy.deepcopy(Yv_p[i][:,1])
    fpr, tpr, p_cutoff = metrics.roc_curve(Yv.iloc[:,i], Yv_p[i][:,1])
    category_auc_rfc.append(roc_auc_score(Yv.iloc[:,i], Yv_p[i][:,1]))
    plt.plot(fpr, tpr, color=colormap[i], label = Yv.columns[i]) 
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

plt.bar(np.arange(len(Yv.columns)), category_auc_rfc, align='center', alpha=0.5)
plt.xticks(np.arange(len(Ys.columns)), Ys.columns)
plt.xticks(rotation=90)
plt.ylabel('AUC')
plt.title('AUC for each category')
plt.show()

plt.bar(np.arange(len(Yv.columns)), Yv.sum(axis=0), align='center', alpha=0.5)
plt.xticks(np.arange(len(Yv.columns)), Yv.columns)
plt.xticks(rotation=90)
plt.ylabel('Labeled Samples')
plt.title('Samples for each category')
plt.show()

 

# Print feature importances
rfc_feature_imp = pd.DataFrame({'Feature': X.columns.values, 'Importance':rfc.feature_importances_}).sort_values('Importance', ascending = False)
print(rfc_feature_imp.head(20))

Xs_test_temp = pd.DataFrame(Xs.loc[Xs_test_index])
Xs_test_temp.reset_index(drop=True, inplace=True)
Ys_test_temp = pd.DataFrame(Ys.loc[Xs_test_index])
Ys_test_temp.reset_index(drop=True, inplace=True)
rfc_test_df_prob = pd.concat([Xs_test_temp, Ys_test_temp, Ys_test_category_prob ], axis=1, ignore_index =True, sort=False)
rfc_test_df_prob.columns = list(["text"]) + list(Ys.columns.values)+ list('pred_'+Ys_test_category_prob.columns.values)



import dill
filename = 'step_2_classification.pkl'
dill.dump_session(filename)
# and to load the session again:
#dill.load_session(filename)




