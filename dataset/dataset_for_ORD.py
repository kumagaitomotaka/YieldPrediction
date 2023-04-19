import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem


ATOM_LIST = list(range(0,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)

    def __getitem__(self, index):
        mol_num = 4 #inputx3+product
        mol_range_list = list(range(len(self.smiles_data)))
        #mol_range_list.remove(index)
        rand_index_list = random.sample(mol_range_list, k = mol_num)
        #rand_index_list.append(index)
        type_idx_set = []
        chirality_idx_set = []
        edge_feat_set = []
        mol_class_set = []
        row_set = []
        col_set = []
        N_sum = 0
        M_sum = 0
        all_atom_num = 0
        for i in range(mol_num):
            rand_index = rand_index_list[i]
            smiles = self.smiles_data[rand_index]
            mol = Chem.MolFromSmiles(smiles)
            # mol = Chem.AddHs(mol)

            N = mol.GetNumAtoms()
            N_sum+=N
            M = mol.GetNumBonds()
            M_sum+=M

            type_idx = []
            chirality_idx = []
            mol_class = []
            #atomic_number = []
            # aromatic = []
            # sp, sp2, sp3, sp3d = [], [], [], []
            # num_hs = []
            atom_num = 0
            for atom in mol.GetAtoms():
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                if i == mol_num-1:
                    mol_class.append(1)
                else:
                    mol_class.append(0)
                atom_num+=1
                #atomic_number.append(atom.GetAtomicNum())
                # aromatic.append(1 if atom.GetIsAromatic() else 0)
                # hybridization = atom.GetHybridization()
                # sp.append(1 if hybridization == HybridizationType.SP else 0)
                # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
            type_idx_set.extend(type_idx)
            chirality_idx_set.extend(chirality_idx)
            mol_class_set.extend(mol_class)
            # z = torch.tensor(atomic_number, dtype=torch.long)
            #x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
            #x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
            #x = torch.cat([x1, x2], dim=-1)
            # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
            #                     dtype=torch.float).t().contiguous()
            # x = torch.cat([x1.to(torch.float), x2], dim=-1)

            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start+all_atom_num, end+all_atom_num]
                col += [end+all_atom_num, start+all_atom_num]
                # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
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
            before_prod_num = all_atom_num-atom_num
            #edge_index = torch.tensor([row, col], dtype=torch.long)
            #edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        x1_set = torch.tensor(type_idx_set, dtype=torch.long).view(-1,1)
        x2_set = torch.tensor(chirality_idx_set, dtype=torch.long).view(-1,1)
        x3_set = torch.tensor(mol_class_set, dtype=torch.long).view(-1,1)
        x = torch.cat([x1_set, x2_set, x3_set], dim=-1) #全分子に含まれるatomの数の合計x3
        edge_index = torch.tensor([row_set, col_set], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat_set), dtype=torch.long)

        # random mask a subgraph of the molecule
        num_mask_nodes = max([1, math.floor(0.25*N_sum)])
        num_mask_edges = max([0, math.floor(0.25*M_sum)])
        mask_nodes_i = random.sample(list(range(N_sum)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N_sum)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M_sum)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M_sum)), num_mask_edges)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        x_i = deepcopy(x)
        for atom_idx in mask_nodes_i:
            if atom_idx > before_prod_num:
                x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST)-1, 0, 1])
            else:
                x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST)-1, 0, 0])
        edge_index_i = torch.zeros((2, 2*(M_sum-num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2*(M_sum-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M_sum):
            if bond_idx not in mask_edges_i:
                edge_index_i[:,count] = edge_index[:,bond_idx]
                edge_attr_i[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            if atom_idx > before_prod_num:
                x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST)-1, 0, 1])
            else:
                x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST)-1, 0, 0])
        edge_index_j = torch.zeros((2, 2*(M_sum-num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2*(M_sum-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M_sum):
            if bond_idx not in mask_edges_j:
                edge_index_j[:,count] = edge_index[:,bond_idx]
                edge_attr_j[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
