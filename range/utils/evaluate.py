import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import top_k_accuracy_score, make_scorer
from scipy.special import softmax


# Custom scoring function: top-k accuracy
def custom_top_k_accuracy(y_true, y_score, k=5, labels=None):
    return top_k_accuracy_score(y_true, y_score, k=k, labels=labels)


def evaluate_npz(args):
    train_path = os.path.join(args.embeddings_dir, args.location_model_name, args.task_name+'_train.npz')
    val_path = os.path.join(args.embeddings_dir, args.location_model_name, args.task_name+'_val.npz')
    assert os.path.exists(train_path), f'Train embeddings file does not exist: {train_path}'
    assert os.path.exists(val_path), f'Val embeddings file does not exist: {val_path}'
    #get training data
    train_data = np.load(train_path, allow_pickle=True)
    train_embeddings = train_data['embeddings']
    train_labels = train_data['y']
    #get validation data
    val_data = np.load(val_path, allow_pickle=True)
    val_embeddings = val_data['embeddings']
    val_labels = val_data['y']

     #decide the model
    if args.task_name == 'ecoregion' or args.task_name == 'biome' or args.task_name == 'country' or args.task_name=='landcover' or 'checker' in args.task_name or args.task_name=='ocean':
        print('Classification Model')
        clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
    elif 'inat' in args.task_name:
        #create the scorer
        print('Top-100 Classification Model')
        top_k_scorer = make_scorer(custom_top_k_accuracy, k=1, labels=np.arange(8142))
        clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=5)
    elif args.task_name == 'nabirds':
        #create the scorer
        print('Top-k Classification Model')
        clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=9, 
        scoring=top_k_scorer)
    else:
        print('Regression Model')
        clf = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=3)
    #normalize the embeddings
    # import code; code.interact(local=dict(globals(), **locals()))
    scaler = MinMaxScaler()

    if args.task_name == 'inat_1':  
        train_inat = np.load('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/inat2018_train_feats.npz')
        train_feats = train_inat['features']
        val_inat = np.load('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/inat2018_val_feats.npz')
        val_feats = val_inat['features']
        train_embeddings = np.concatenate([train_embeddings, train_feats], axis=1)
        val_embeddings = np.concatenate([val_embeddings, val_feats], axis=1)
        train_embeddings = scaler.fit_transform(train_embeddings)
        val_embeddings = scaler.transform(val_embeddings)
        # import code; code.interact(local=dict(globals(), **locals()))
        #run the classifier
        clf.fit(train_embeddings, train_labels)
        val_predictions = clf.decision_function(val_embeddings)
        final_prob = softmax(val_predictions, axis=1)
        val_accuracy_1 = top_k_accuracy_score(val_labels, final_prob, k=1, labels=np.arange(8142))
        val_accuracy_3 = top_k_accuracy_score(val_labels, final_prob, k=3, labels=np.arange(8142))
        val_accuracy_5 = top_k_accuracy_score(val_labels, final_prob, k=5, labels=np.arange(8142))
        print(f'Top-1 accuracy is {val_accuracy_1}')
        print(f'Top-3 accuracy is {val_accuracy_3}')
        print(f'Top-5 accuracy is {val_accuracy_5}')
        val_accuracy = val_accuracy_1
    else:
        train_embeddings = scaler.fit_transform(train_embeddings)
        val_embeddings = scaler.transform(val_embeddings)
        #run the classifier
        clf.fit(train_embeddings, train_labels)
        val_accuracy = clf.score(val_embeddings, val_labels)
        print(f'The validation set accuracy is {val_accuracy:3f}')
    return val_accuracy