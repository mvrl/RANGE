import os
#ML imports
import torch
from torch.utils.data import random_split, DataLoader
#local import of datasets
from ..evaluation.evaldatasets import Biome_Dataset, Eco_Dataset, Temp_Dataset, Housing_Dataset, Elevation_Dataset, Population_Dataset, Country_Dataset, Inat_Dataset, CSVDataset, CheckerDataset, Ocean_Dataset, ERA5_Dataset 


def get_dataset(args):
    generator = torch.Generator().manual_seed(42)
    if args.task_name == 'biome':
        data_path = args.eval_dir
        dataset = Biome_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'ecoregion':
        data_path = args.eval_dir
        dataset = Eco_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'country':
        data_path = os.path.join(args.eval_dir,'country.csv')
        dataset = Country_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name=='ocean':
        train_data_path = os.path.join(args.eval_dir,'land_ocean_train.csv')
        test_data_path = os.path.join(args.eval_dir, 'land_ocean_test.csv')
        dataset_train = Ocean_Dataset(train_data_path)
        dataset_val = Ocean_Dataset(test_data_path)
        num_classes = dataset_train.num_classes

    elif args.task_name == 'temperature':
        data_path = os.path.join(args.eval_dir, 'temp.csv')
        dataset = Temp_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'housing':
        data_path = os.path.join(args.eval_dir, 'housing.csv')
        dataset = Housing_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'elevation':
        data_path = os.path.join(args.eval_dir, 'elevation.csv')
        dataset = Elevation_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'population':
        data_path = os.path.join(args.eval_dir, 'population.csv')
        dataset = Population_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes

    elif args.task_name == 'inat_1':
        data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/'
        dataset_train = Inat_Dataset(data_path, type='train')
        dataset_val = Inat_Dataset(data_path, type='val')
        num_classes = dataset_train.num_classes

    elif args.task_name == 'csv_data':
        data_path = os.path.join(args.eval_dir,'/cont_haver.csv')
        dataset_train = CSVDataset(data_path)
        dataset_val  = CSVDataset(data_path)
        num_classes = dataset_train.num_classes
    elif 'era5' in args.task_name:
        data_path = os.path.join(args.eval_dir, 'ERA5_Land_Clipped_2020.csv')
        group = args.task_name.split('-')[-1]
        dataset = ERA5_Dataset(data_path, group)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset.num_classes 
    elif 'checker' in args.task_name:
        num_support = int(args.task_name.split('_')[-1])
        num_classes = 16
        ds = CheckerDataset(num_samples=10000, num_classes=num_classes, num_support=num_support)
        dataset_train = ds.train_ds
        dataset_val = ds.evalu_ds
        num_classes = num_classes
    else:
        raise ValueError('Task name not recognized')

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    return train_loader, val_loader, num_classes