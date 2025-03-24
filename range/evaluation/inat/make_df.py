import numpy as np
import pandas as pd
from .datasets import load_dataset

if __name__ == '__main__':
      import code; code.interact(local=dict(globals(), **locals()))
      params = {'dataset':'inat_2018', 'load_img':False,
            'cnn_pred_type':'full', 'cnn_model':'inception_v3', 'inat2018_resolution':'other'}
      eval_type='val'
      dataset = load_dataset(params, eval_type, load_cnn_features=False, load_cnn_features_train=False)
      import code; code.interact(local=dict(globals(), **locals()))
      dataset['train_lons'] = dataset['train_locs'][:,0]
      dataset['train_lats'] = dataset['train_locs'][:,1]
      train_dict = {}
      train_dict['lon'] = dataset['train_lons']
      train_dict['lat'] = dataset['train_lats']
      train_dict['class'] = dataset['train_classes']
      train_df = pd.DataFrame(train_dict)
      np.savez('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/inat2018_train_feats.npz',
               lat=dataset['train_lats'], lon=dataset['train_lons'], classes=dataset['train_classes'],
               features=dataset['train_feats'])
    
      dataset['val_lons'] = dataset['val_locs'][:,0]
      dataset['val_lats'] = dataset['val_locs'][:,1]
      val_dict = {}
      val_dict['lon'] = dataset['val_lons']
      val_dict['lat'] = dataset['val_lats']
      val_dict['class'] = dataset['val_classes']
      val_dict['prediction'] = dataset['val_preds'].tolist()
      df = pd.DataFrame(val_dict)
      np.savez('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/inat2018_val_feats_inception.npz',
               lat=dataset['val_lats'], lon=dataset['val_lons'], classes=dataset['val_classes'],
               prediction=dataset['val_preds'])
      import code; code.interact(local=dict(globals(), **locals()))