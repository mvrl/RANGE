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
    if args.task_name == 'ecoregion' or args.task_name == 'biome' or args.task_name == 'country'  or 'checker' in args.task_name or args.task_name=='ocean':
        print('Classification Model')
        clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
    elif 'inat' in args.task_name:
        raise NotImplementedError('Inat evaluation not implemented')
    else:
        print('Regression Model')
        clf = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=3)
    #normalize the embeddings
    # import code; code.interact(local=dict(globals(), **locals()))
    scaler = MinMaxScaler()


    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)
    #run the classifier
    clf.fit(train_embeddings, train_labels)
    val_accuracy = clf.score(val_embeddings, val_labels)
    print(f'The validation set accuracy is {val_accuracy:3f}')
    return val_accuracy