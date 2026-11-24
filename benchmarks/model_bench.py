"""5-fold comparison of fallback text classifiers on unseen message templates (needs scikit-learn). Run: python benchmarks/model_bench.py"""
import sys,time,json,collections
import os;ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(ROOT,'src'))
import numpy as np, scipy.sparse as sp
from error_classifier import build_training_set,_WORD_TOKEN_RE
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score
D=os.path.join(ROOT,'data','training_data.json')
T,L,st=build_training_set(D); T=np.array(T,dtype=object);L=np.array(L,dtype=object)
print(len(T),len(set(L)),st['classes'])
def feats(kind):
    def f(tr,te):
        parts=[]
        if kind in('both','word'):
            wv=TfidfVectorizer(ngram_range=(1,2),token_pattern=_WORD_TOKEN_RE.pattern,dtype=np.float32);parts.append((wv.fit_transform(tr),wv.transform(te)))
        if kind in('both','char'):
            cv=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,dtype=np.float32);parts.append((cv.fit_transform(tr),cv.transform(te)))
        if kind=='count':
            cv=CountVectorizer(ngram_range=(1,2),token_pattern=_WORD_TOKEN_RE.pattern);parts.append((cv.fit_transform(tr),cv.transform(te)))
        return sp.hstack([a for a,_ in parts]).tocsr(),sp.hstack([b for _,b in parts]).tocsr()
    return f
M={
 'TF-IDF + LogReg (ours)':('both',lambda:LogisticRegression(max_iter=3000,C=20,class_weight='balanced')),
 'TF-IDF + Linear SVM':('both',lambda:LinearSVC(C=1.0,class_weight='balanced')),
 'Word TF-IDF + LogReg':('word',lambda:LogisticRegression(max_iter=3000,C=20,class_weight='balanced')),
 'Char TF-IDF + LogReg':('char',lambda:LogisticRegression(max_iter=3000,C=20,class_weight='balanced')),
 'BoW + Naive Bayes':('count',lambda:MultinomialNB(alpha=0.1)),
 'TF-IDF + kNN (k=3)':('both',lambda:KNeighborsClassifier(3,metric='cosine',algorithm='brute')),
 'TF-IDF + Random Forest':('both',lambda:RandomForestClassifier(200,n_jobs=-1,random_state=0)),
}
kf=KFold(5,shuffle=True,random_state=42);res={}
for name,(k,mk) in M.items():
    acc=[];f1=[];tt=[];lat=[]
    for tr,te in kf.split(T):
        Xa,Xb=feats(k)(list(T[tr]),list(T[te]))
        m=mk();t=time.time();m.fit(Xa,L[tr]);tt.append(time.time()-t)
        t=time.time();p=m.predict(Xb);lat.append((time.time()-t)/len(te)*1000)
        acc.append(accuracy_score(L[te],p));f1.append(f1_score(L[te],p,average='macro'))
    res[name]=dict(acc=float(np.mean(acc)),acc_sd=float(np.std(acc)),f1=float(np.mean(f1)),train_s=float(np.mean(tt)),ms_per_msg=float(np.mean(lat)))
    print(name,res[name],flush=True)
json.dump(dict(n=len(T),classes=len(set(L)),folds=5,res=res),open('model_bench_results.json','w'),indent=1)
