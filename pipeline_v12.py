# Developed by Ben Sinclair and Jarrel Seah, Momnash University
# Last updated 28/08/2021


from __future__ import division

#================================================
# Clear variables
#================================================
from IPython import get_ipython
get_ipython().magic('reset -sf')

#================================================
# Importing the libraries
#================================================
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTENC
import sklearn
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
#import xgboost
from scipy.stats import uniform
import numpy as np
import os
import sys
import pickle
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import linear_model
import itertools
import pickle
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from scipy import stats

sys.path.append(r'/Users/bsin0008/Documents/Work/Monash/Code/ProjectSO_SurgicalOutcomesPET/sklearn')

class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)
        
    def __call__(self, res):
        self._bar.update()


# In[39]:

class SmoteStratifiedKFold(StratifiedKFold):
    def __init__(self, n_splits=5, num_smote=0, *, shuffle=False, random_state=None):        
        super().__init__(n_splits=n_splits, shuffle=shuffle,random_state=None)
        self.num_smote = num_smote
        
    def split(self, X, y, groups=None):
        X_orig = X[:-self.num_smote]
        y_orig = y[:-self.num_smote]
        smoted_indices = list(range(len(X)-self.num_smote, len(X)))
        ret = list(super().split(X_orig, y_orig))       
        for i, ds in enumerate(ret):            
            ds = list(ds)
            ds[0] = np.array(list(ds[0]) + smoted_indices)
            ret[i] = ds
        return ret            


def twoLayerFeedForward(l2val=0.001):
    clf = Sequential()
    clf.add(Dense(7, activation='relu',kernel_initializer = 'uniform',kernel_regularizer=l2(l2val),bias_regularizer=l2(l2val)))
    clf.add(Dense(7, activation='relu',kernel_initializer = 'uniform',kernel_regularizer=l2(l2val),bias_regularizer=l2(l2val)))
    clf.add(Dense(1, activation='sigmoid',kernel_initializer = 'uniform',kernel_regularizer=l2(l2val)))
    #clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    clf.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return clf


#turn off warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None  # default='warn'

time0 = time.time()


        
#================================================
#================================================
# Options
#================================================
#================================================
options = {}
options['pipeline'] = {}
options['pipelineparams'] = {}
options['params'] = {}
options['files'] = {}  

#=======================================
# Data
#=======================================
indir='/Users/bsin0008/Documents/Work/Monash/Subject_Info/ProjectSO_SurgicalOutcomes/'
#outdir='/Users/bsin0008/Documents/Work/Monash/Results/ProjectSO_SurgicalOutcomes/sklearn_classification_results/version7/'
outdir='/Users/bsin0008/Documents/Work/Monash/Results/ProjectSO_SurgicalOutcomes/sklearn_classification_results/version11/'
#inname='Sz_analysis_TLd3_for_BS_13.05.18_szburden_reduced_anon.csv'
inname='Sz_analysis_TLd3_for_BS_13.05.18_szburden.csv'
options['files']['infname']=os.path.join(indir,inname) 


#=======================================
# Pipeline
#=======================================
#options['pipeline']['crossval']="StratifiedShuffleSplit"
#options['pipeline']['crossval']="StratifiedKFold"
options['pipeline']['crossval']="RepeatedStratifiedKFold"
options['pipelineparams']['n_repeats']=10
options['pipelineparams']['n_folds']=10

options['pipeline']['do_smote']=1

#options['pipeline']['hpopt']='none'
#options['pipeline']['hpopt']='GridSearchCV'
options['pipeline']['hpopt']='BayesSearchCV'; 
n_iter_bs=30
n_splits_hp=5

options['pipeline']['statcomp']="paired_ttest_NBcorrected"
options['pipeline']['comparison']="algos"
#options['pipeline']['comparison']="ATL"
#options['pipeline']['comparison']="surgical"

#=======================================
# Steps
#=======================================
#options['pipeline']['steps']=["ttest"]
options['pipeline']['steps']=["fit","fitresults"]
#options['pipeline']['steps']=["fitresults"]
#options['pipeline']['steps']=["statistical_comparison"]

options['pipeline']['do_print_fold_stats']=0
options['pipeline']['do_print_hpopt_stats']=0
options['pipeline']['do_save_scores']=1


#=======================================
# Model
#=======================================
modelparams = []
modelparams.append(['LR',LogisticRegression(tol=1e-2, max_iter=200,random_state=0),{'C': (0.1, 10.0, 'uniform')}])
#modelparams.append(['SVM',SVC(random_state=0, probability=True),{'C':(1,10,'uniform')}])
#modelparams.append(['RF',RandomForestClassifier(random_state=0),{'n_estimators':(3,100, 'uniform'),'max_depth':(3,30, 'uniform'),'max_features':['sqrt', None]}])
#modelparams.append(['ANN',MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7, 7), random_state=0),{'alpha':(0.00001,0.001,'uniform')}])
modelANNk = KerasClassifier(twoLayerFeedForward, batch_size=9999, epochs=100,verbose=0 )
#modelparams.append(['ANNk',modelANNk,{'l2val':(0.00001,0.01, 'uniform')}])
           
 
#=======================================
# Variables
#=======================================
vars_dep = ['Szoutcomeatlastfollowup']

vars_indep_prepost_TL       = ['MRIcode','ctl_TL_wbmincbm_mask','pc_ET_hypo_preop','vol_preop_TL_hypo_mm3','tissueresected','pc_TLhypo_resected'] # "prepost" TL ### CAHILL
vars_indep_pre              = ['MRIcode','ctl_TL_wbmincbm_mask','pc_ET_hypo_preop','vol_preop_TL_hypo_mm3'] #"pre"

#ATL
vars_indep_prepost_TL_ATL   = ['MRIcode','ctl_TL_wbmincbm_mask','ATLcode','pc_ET_hypo_preop','vol_preop_TL_hypo_mm3','tissueresected','pc_TLhypo_resected'] # "prepost" TL ### CAHILL
vars_indep_pre_ATL          = ['MRIcode','ctl_TL_wbmincbm_mask','ATLcode','pc_ET_hypo_preop','vol_preop_TL_hypo_mm3'] #"pre"

vars_indeps = []
#vars_indeps.append(vars_indep_prepost_TL)
vars_indeps.append(vars_indep_prepost_TL_ATL)

vars_cat = ['MRIcode', 'ctl_TL_wbmincbm_mask', 'ATLcode','gendercode']
vars_cont = ['vol_preop_TL_hypo_mm3','pc_ET_hypo_preop','tissueresected','pc_TLhypo_resected']
   
#=======================================
# Cross Validation
#=======================================
if options['pipeline']['crossval']=="KFold":
    cv = KFold(n_splits=options['pipelineparams']['n_folds'],random_state=0)  
elif options['pipeline']['crossval']=="StratifiedShuffleSplit": 
    cv = StratifiedShuffleSplit(n_splits=options['pipelineparams']['n_repeats'], test_size=0.2, random_state=0)        
elif options['pipeline']['crossval']=="StratifiedKFold":    
    cv = StratifiedKFold(n_splits=options['pipelineparams']['n_folds'],random_state=0)
elif options['pipeline']['crossval']=="RepeatedStratifiedKFold":   
    cv = RepeatedStratifiedKFold(n_splits=options['pipelineparams']['n_folds'],n_repeats=options['pipelineparams']['n_repeats'],random_state=0)

#=======================================
# Load Data
#=======================================  
df = pd.read_csv(options['files']['infname'])

    
#=======================================
#  Loop over Variables
#=======================================
for vars_indep in vars_indeps: 

    vars_all=vars_indep.copy(); 
    vars_all.append(vars_dep[0])
    vars_indep_cat = [i for i in vars_indep if i in vars_cat] 
    vars_indep_cont = [i for i in vars_indep if i in vars_cont] 
    vars_indep_cat_inds=np.array([vars_indep.index(i) for i in vars_indep_cat])
    df_used = df[vars_all]   
    
    
    variables_str='-'.join(vars_indep)
    methodstr=('%s' % options['pipeline']['crossval'])+('%i' % options['pipelineparams']['n_repeats'] + 'x' + ('%i' % options['pipelineparams']['n_folds']))
    outfname={}
    
    if "ttest" in options['pipeline']['steps']: #NB, only works if ATLcode is amongst variables
    
        yraw = df[vars_dep].values    
        inds_E1 = np.where(yraw==1); inds_E1=np.asarray(inds_E1); inds_E1=inds_E1[0,:]
        inds_E2 = np.where(yraw==2); inds_E2=np.asarray(inds_E2); inds_E2=inds_E2[0,:]

        for i_var in vars_indep:
            var_E1 = df[i_var].values[inds_E1]
            var_E2 = df[i_var].values[inds_E2]  
            
            if np.isin(i_var,vars_cat):
                fetable = np.zeros([2,2])
                unique, counts1 = np.unique(var_E1, return_counts=True)
                unique, counts2 = np.unique(var_E2, return_counts=True)
                fetable[0,0] = counts1[0]; fetable[0,1] = counts1[1]
                fetable[1,0] = counts2[0]; fetable[1,1] = counts2[1]
                t2, p2 = stats.fisher_exact(fetable,alternative="two-sided")  
                print("%s:\t%i/%i (%3.1f)\t%i/%i (%3.1f)\t%5.3f" % (i_var,counts1[0],counts1[0]+counts1[1],100*counts1[0]/(counts1[0]+counts1[1]),counts2[0],counts2[0]+counts2[1],100*counts2[0]/(counts2[0]+counts2[1]),p2))
            else:    
                t2, p2 = stats.mannwhitneyu(var_E1,var_E2,alternative="two-sided")
                print("%s:\t%5.1f (%5.1f-%5.1f)\t%5.1f (%5.1f-%5.1f)\t%5.3f" % (i_var,np.median(var_E1),np.quantile(var_E1,0.25),np.quantile(var_E1,0.75),np.median(var_E2),np.quantile(var_E2,0.25),np.quantile(var_E2,0.75),p2))
            
    
    for modelname, model, params in modelparams:
        outname = 'classifications_' + modelname +'_' + variables_str + '_' + methodstr + '.pkl'
        outfname[modelname]=os.path.join(outdir,outname)
    
        #=======================================
        # Fit Model
        #=======================================
        if "fit" in options['pipeline']['steps']: 
            
            #================================================
            # Preprocessing
            #================================================
            df_used.Szoutcomeatlastfollowup -= 1
            df_used.MRIcode -= 1
            df_used.ctl_TL_wbmincbm_mask -= 1
            
            sc = StandardScaler()
            for i_var in vars_indep_cont:
                df_used[i_var] = sc.fit_transform(df_used[i_var].values.reshape(-1, 1))
            
            X = df_used[vars_indep].values
            y = df_used[vars_dep].values 
            
            
            #=======================================
            # Cross Validation for hpopt
            #=======================================
            if options['pipeline']['do_smote']==1:             
                cv_hp = SmoteStratifiedKFold(n_splits=n_splits_hp, num_smote=9999, random_state=0) 
            else:
                cv_hp = StratifiedKFold(n_splits=n_splits_hp,random_state=0)
            
             
            #=======================================
            # Loop over models
            #=======================================        
            modelscores = []
            for modelname, model, params in modelparams:
                
                foldscores = []
                foldbestest = []
                
                if options['pipeline']['hpopt']=='none':
                    clf = model
                elif options['pipeline']['hpopt']=='GridSearchCV':   
                    clf = GridSearchCV(model, params, cv=cv_hp, verbose=0, refit=True, scoring='neg_log_loss')
                elif options['pipeline']['hpopt']=='BayesSearchCV':
                    clf = BayesSearchCV(model, params, cv=cv_hp, verbose=0, refit=True, scoring='neg_log_loss', random_state=0, n_iter=n_iter_bs)
                
                #=======================================
                # Loop through Cross Validation splits
                #=======================================
                i_repeat=0
                 
                for train_index, test_index in cv.split(X, y):
                    
                    scores = []
                    i_repeat=i_repeat+1
                    print('repeat: ',i_repeat)
                    time0repeat = time.time()
                    
                    train_df = df_used.loc[train_index]
                    test_df = df_used.loc[test_index]         
                      
                    # get train X and y
                    if options['pipeline']['do_smote']==1: 
                        smote_nc = SMOTENC(categorical_features=vars_indep_cat_inds,random_state=0)
                        X_fit, y_fit = smote_nc.fit_resample(train_df[vars_indep].values, train_df['Szoutcomeatlastfollowup'].values)   
                        clf.cv.num_smote=(len(y_fit) - len(train_df.values))  
                    else:
                        X_fit = train_df[vars_indep].values
                        y_fit = train_df['Szoutcomeatlastfollowup'].values
                        
                    # fit model    
                    if modelname=="ANNk":     
                        #kwargs={"batch_size": X_fit.shape[0]}
                        clf.estimator.set_params(**{'batch_size': X_fit.shape[0]}) # set batch size to n_train
                        clf.fit(X_fit, y_fit)    # batch gradient descent
                    else:
                        clf.fit(X_fit, y_fit)
                    
                    # Get predictions
                    if options['pipeline']['hpopt']=='none':
                        predicted = clf.predict(test_df[vars_indep].values) 
                        predictedproba = clf.predict_proba(test_df[vars_indep].values) 
                    else:    
                        predicted = clf.best_estimator_.predict(test_df[vars_indep].values) # predictions of BEST estimator
                        predictedproba = clf.best_estimator_.predict_proba(test_df[vars_indep].values) # predictions of BEST estimator
                    
                    if predictedproba.shape[1]==1:
                        predictedproba_1=predictedproba
                    elif predictedproba.shape[1]==2:    
                        predictedproba_1=predictedproba[:,1]
                        
                    predictedproba_bin = np.array(predictedproba_1>0.5,dtype=np.uint8) 
                    
                    # Get performance measures
                    auc_test  = sklearn.metrics.roc_auc_score(test_df.Szoutcomeatlastfollowup.values, predictedproba_1)
                    acc_test = sklearn.metrics.accuracy_score(test_df.Szoutcomeatlastfollowup.values, predicted)
                    #####from confusion matrix calculate accuracy
                    cm_test = confusion_matrix(test_df.Szoutcomeatlastfollowup.values, predicted)     
                    #acm_test = ( cm_test[0,0]+cm_test[1,1] ) / ( cm_test[0,0]+cm_test[0,1]+cm_test[1,0]+cm_test[1,1] )       
                    sen_test = ( cm_test[0,0] ) / ( cm_test[0,0]+cm_test[0,1] )
                    spc_test = ( cm_test[1,1] ) / ( cm_test[1,0]+cm_test[1,1] )
                    ppv_test = ( cm_test[0,0] ) / ( cm_test[0,0]+cm_test[1,0] )
                    npv_test = ( cm_test[1,1] ) / ( cm_test[0,1]+cm_test[1,1] )
            
                    
                    scores.append(auc_test)
                    scores.append(acc_test)
                    #scores.append(acm_test)
                    scores.append(sen_test)
                    scores.append(spc_test)
                    scores.append(ppv_test)
                    scores.append(npv_test)
                    
                    if options['pipeline']['do_print_fold_stats']==1: 
                        
                        print(vars_indep)
                        print('model: ',modelname)                      
                        
                        if options['pipeline']['do_print_hpopt_stats']==1: 
                            print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
                            search_means = clf.cv_results_['mean_test_score']
                            search_stds = clf.cv_results_['std_test_score']
                            search_params = clf.cv_results_['params']
                            #for search_mean, search_stdev, search_param in zip(search_means, search_stds, search_params):
                            #    print("%f (%f) with: %r" % (search_mean, search_stdev, search_param))
            
                        #print(clf.best_params_)
                        print('auc: ',auc_test)
                        print('acc: ',acc_test)
                        #print(clf.best_estimator_,', acm: ',acm_test)
                        print('sen: ',sen_test)
                        print('spc: ',spc_test)
                        print('ppv: ',ppv_test)
                        print('cpv: ',npv_test)
                    
                    foldbestest=clf
                    foldscores.append(scores)
                    
                    elapsedrepeat = time.time() - time0repeat  
                    print('elapsed repeat: ',elapsedrepeat)
                    
                    ### END loop over folds
                    
                modelscores.append(foldscores)
                
                if options['pipeline']['do_save_scores']==1:         
                    with open(outfname[modelname], 'wb') as fname:  # Python 3: open(..., 'wb')
                        if modelname=='ANNk': # cant pickle keras wrapper weakref object
                            model='dummy' 
                            clf='dummy'
                        pickle.dump([modelname, model, params, vars_indep, foldscores, options, clf], fname)  
             
                modelname_, model_, params_, vars_indep_, foldscores_, options_, clf_ = modelname, model, params, vars_indep, foldscores, options, clf 
                
            
        else: # if "fit" in options['pipeline']['steps']: 
            
            with open(outfname[modelname],"rb") as fname:  # Python 3: open(..., 'wb')
                    modelname_, model_, params_, vars_indep_, foldscores_, options_, clf_  = pickle.load(fname)  
            
        #=======================================
        # Print performance measures
        #=======================================    
        if "fitresults" in options['pipeline']['steps']: 
                
            print(vars_indep_)    
            print(modelname_)
            #scoresall=pd.DataFrame(foldscores_)
            scoresall=np.array(foldscores_)
            #print(scoresall.describe())
            scoresmean=np.nanmean(scoresall,0)
            # print(scoresmean)
            for iscore in scoresmean:
                print('%.2f' % iscore)
    
        ### END loop over models
        
### END loop over variables

  

    
#=======================================
# Statistical Comparison
#=======================================
if "statistical_comparison" in options['pipeline']['steps']:

    from compare_models import compare_models_loaded    
            
    modelname1_list=[]
    modelname2_list=[]
    vars_indep1_list=[]
    vars_indep2_list=[]
    
    #5a) SVM vs LR  
    if options['pipeline']['comparison']=="algos":  
           
        for vars_indep in vars_indeps:
   
            modelname1_list.append('LR')
            vars_indep1_list.append(vars_indep)
            modelname2_list.append('SVM')
            vars_indep2_list.append(vars_indep)
        
            modelname1_list.append('LR')
            vars_indep1_list.append(vars_indep)
            modelname2_list.append('RF')
            vars_indep2_list.append(vars_indep)    
            
            modelname1_list.append('LR')
            vars_indep1_list.append(vars_indep)
            modelname2_list.append('ANNk')
            vars_indep2_list.append(vars_indep)  
                                    
                
    #5c) ATLR vs no ATLR 
    elif options['pipeline']['comparison']=="ATL":        
          
        for modelname in ['LR', 'SVM', 'RF', 'ANNk']:
            
            modelname1_list.append(modelname)
            vars_indep1_list.append(vars_indep_prepost_TL)
            modelname2_list.append(modelname)
            vars_indep2_list.append(vars_indep_prepost_TL_ATL)

    
    #5b) prepost vs pre               
    elif options['pipeline']['comparison']=="surgical":
        
        for modelname in ['LR', 'SVM', 'RF', 'ANNk']:

            modelname1_list.append(modelname)
            vars_indep1_list.append(vars_indep_pre_ATL)
            modelname2_list.append(modelname)
            vars_indep2_list.append(vars_indep_prepost_TL_ATL)

                
    for modelname1, modelname2, vars_indep1, vars_indep2 in zip(modelname1_list , modelname2_list, vars_indep1_list, vars_indep2_list):
  
        variables_str1='-'.join(vars_indep1)
        variables_str2='-'.join(vars_indep2)
        methodstr1=('%s' % options['pipeline']['crossval'])+('%i' % options['pipelineparams']['n_repeats'] + 'x' + ('%i' % options['pipelineparams']['n_folds']))
        methodstr2=('%s' % options['pipeline']['crossval'])+('%i' % options['pipelineparams']['n_repeats'] + 'x' + ('%i' % options['pipelineparams']['n_folds']))          
        outname1 = 'classifications_' + modelname1 +'_' + variables_str1 + '_' + methodstr1 + '.pkl'
        outname2 = 'classifications_' + modelname2 +'_' + variables_str2 + '_' + methodstr2 + '.pkl'         
        fname1=os.path.join(outdir,outname1)
        fname2=os.path.join(outdir,outname2)
        
        score_t,score_p = compare_models_loaded(fname1,fname2,options)
        print(outname1,'\n','vs','\n',outname2)
        for scoret ,scorep in zip(score_t,score_p):
            print('%.2f\t%.3f' % (scoret, scorep))


    # do FDR correction
    from statsmodels.stats import multitest
    pvals_models_TL_ATL=np.array([0.262,0.203,0.352,0.195,0.172,0.057,0.04,0.42,0.374,0.639,0.936,0.094])
    pvals_post_TL=np.array([0.739,0.585,0.539,0.528,0.615,0.675,0.788,0.368,0.498,0.41,0.289,0.636,0.486,0.455,0.427,0.5]) 
    pvals_post_TL_ATL=np.array([0.031,0.057,0.098,0.299,0.000,0.037,0.330,0.001,0.034,0.005,0.038,0.021,0.006,0.013,0.097,0.077])
    p_fdr_corr=multitest.fdrcorrection(pvals_post_TL_ATL, alpha=0.05, method='indep', is_sorted=False)
    print(p_fdr_corr)
    
elapsed = time.time() - time0  
print('elapsed: ',elapsed)
    
