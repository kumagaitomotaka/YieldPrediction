import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(0,119)) #0番目の原子を追加した
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles_for_ORD(data_path, target, task, rxn_type, target_rxn=None):
    labels = []
    target_labels = []
    smiles_list = []
    target_smiles_list = []
    smiles_name_list = ['smiles0','smiles1','smiles2','product_smiles']
    if target_rxn != None:
        if target_rxn not in rxn_type:
            raise ValueError('target_rxn must be incuded in rxn_type') 
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i >= 0:
                rxn_set = []
                if rxn_type != None:
                    if row['reaction_type'] not in rxn_type:
                        continue
                for n,name in enumerate(smiles_name_list):
                    smiles = row[name]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol != None:
                        rxn_set.append(smiles)
                    else:
                        break #'NoData'などmolオブジェクトに変換できない分子を含むrxnを取り除く(120700/535128)
                label = row[target]
                if len(rxn_set) == len(smiles_name_list) and label != '':
                    if target_rxn != None and row['reaction_type'] == target_rxn:
                        target_smiles_list.append(rxn_set)
                        if task == 'classification':
                            target_labels.append(int(float(label)//10)) #収率を1/10にして0~10の範囲に
                        elif task == 'regression':
                            target_labels.append(float(label/100))
                        else:
                            raise ValueError('task must be either regression or classification') 
                    else:
                        smiles_list.append(rxn_set)
                        if task == 'classification':
                            labels.append(int(float(label)//10)) #収率を1/10にして0~10の範囲に
                        elif task == 'regression':
                            labels.append(float(label/100))
                        else:
                            raise ValueError('task must be either regression or classification')
    if target_rxn != None:
        print('target_smiles_list({}): '.format(target_rxn), len(target_smiles_list))
        print('smiles_list: ', len(smiles_list))
        print('target_labels({}): '.format(target_rxn), len(target_labels))
        print('labels: ', len(labels))
        return smiles_list, labels, target_smiles_list, target_labels
    else:
        print('smiles_list: ', len(smiles_list))
        print('labels: ', len(labels))
        return smiles_list, labels


class MolTestDataset_for_ORD(Dataset):
    def __init__(self, data_path, target, task, rxn_type, smiles_set=None):
        super(Dataset, self).__init__()
        self.task = task
        if smiles_set != None:
            self.smiles_list = smiles_set[0]
            self.labels = smiles_set[1]
        else:
            self.smiles_list, self.labels = read_smiles_for_ORD(data_path, target, task, rxn_type)

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        type_idx_set = []
        chirality_idx_set = []
        edge_feat_set = []
        mol_class_set = []
        row_set = []
        col_set = []
        all_atom_num = 0
        smiles_set = self.smiles_list[index]
        for i in range(len(smiles_set)):
            mol = Chem.MolFromSmiles(smiles_set[i])
            mol = Chem.AddHs(mol)

            N = mol.GetNumAtoms()
            M = mol.GetNumBonds()

            type_idx = []
            chirality_idx = []
            mol_class = []
            #atomic_number = []
            atom_num = 0
            for atom in mol.GetAtoms():
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                if i == len(smiles_set)-1:
                    mol_class.append(1)
                else:
                    mol_class.append(0)
                #atomic_number.append(atom.GetAtomicNum())
                atom_num+=1

            type_idx_set.extend(type_idx)
            chirality_idx_set.extend(chirality_idx)
            mol_class_set.extend(mol_class)
            #x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
            #x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
            #x = torch.cat([x1, x2], dim=-1)
            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start+all_atom_num, end+all_atom_num]
                col += [end+all_atom_num, start+all_atom_num]
                edge_feat.append([
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir())
                ])
                edge_feat.append([
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir())
                ])
            row_set.extend(row)
            col_set.extend(col)
            edge_feat_set.extend(edge_feat)
            all_atom_num+=atom_num
            #edge_index = torch.tensor([row, col], dtype=torch.long)
            #edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        x1_set = torch.tensor(type_idx_set, dtype=torch.long).view(-1,1)
        x2_set = torch.tensor(chirality_idx_set, dtype=torch.long).view(-1,1)
        x3_set = torch.tensor(mol_class_set, dtype=torch.long).view(-1,1)
        x = torch.cat([x1_set, x2_set, x3_set], dim=-1) #全分子に含まれるatomの数の合計x3
        edge_index = torch.tensor([row_set, col_set], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat_set), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_list)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting, rxn_type=None, smiles_set=None, dataset_type=None
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.smiles_set = smiles_set
        if dataset_type == 'train_val':
            self.valid_size = valid_size
            self.test_size = 0
        elif dataset_type == 'test':
            self.valid_size = 0
            self.test_size = 1
        else:
            self.valid_size = valid_size
            self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.rxn_type = rxn_type
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        if self.smiles_set != None:
            train_dataset = MolTestDataset_for_ORD(data_path=self.data_path, target=self.target, task=self.task, rxn_type=self.rxn_type, smiles_set=self.smiles_set)
        else:
            train_dataset = MolTestDataset_for_ORD(data_path=self.data_path, target=self.target, task=self.task, rxn_type=self.rxn_type)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            
            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
