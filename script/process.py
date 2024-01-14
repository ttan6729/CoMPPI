#process file, costruct knowledge information
#store the previous script, e.g. below is the scripts for the file that split one sequence
#into multiple lines

import sys
import os
import numpy as np
import pandas as pd
import tqdm #for print the progress note
import networkx as nx
from collections import defaultdict

#reaction activation catalysis binding inhibition ptmod expression
#a protein sequence
class seq:
	uId = "" #uniProtein database ID
	sId = "" #STRING database ID
	seq = ""
	vec = ""
	def __init__(self,_uId=None,_sId=None,_seq=None):
		self.uId = _uId 
		self.sId = _sId 
		self.seq = _seq.strip() # remove \n from the original sequence

#for the csv/tsv file with the following format
# 'item_id_a', 'item_id_b', 'mode', 'action','is_directional', 'a_is_acting', 'score'
def construct_KG(fName): 
	df = None
	if ".csv" in fName:
		print(f'read csv file {fName}, first row used as column name')
		df = pd.read_csv(fName,index_col=False)
	elif ".tsv" in fName:	
		print(f'read tsv file {fName}, first row used as column name')	
		df = pd.read_table(fName,index_col=False)

	mode_alphabet = {'reaction':0,'binding':1,'ptmod':2,'activation':3,'inhibition':4,
	'catalysis':5,'expression':6}
	protein_dict = {}
	KG = defaultdict(list)
	extract_indices = [0,1,2]
	for line in df.values:
		item_a, item_b, mode = line[extract_indices]
		
		if item_a not in protein_dict:
			protein_dict[item_a] = len(protein_dict)
		if item_b not in protein_dict:
			protein_dict[item_b] = len(protein_dict)
		if mode not in mode_alphabet:
			mode_alphabet[mode] = len(mode_alphabet)
			print(f'warning, mode {moed} is not defined, which may affect the performance of method')

		KG[protein_dict[item_a]].append((protein_dict[item_b],mode_alphabet[mode]))
		KG[protein_dict[item_b]].append((protein_dict[item_a],mode_alphabet[mode]))
		
	sample_size = 10 #the maximum number of counted neighobour of each node, need to be tested afterwards
	protein_num = len(protein_dict)

	adj_protein = np.zeros(shape=(protein_num,sample_size),dtype=np.int64)
	adj_inter   = np.zeros(shape=(protein_num,sample_size),dtype=np.int64)

	for i in range(protein_num):
		all_neighbors = KG[i] #all the relationships
		neigh_num = len(all_neighbors)
		sample_ids = np.random.choice(neigh_num,sample_size,
			replace=False if neigh_num >= sample_size else True)

		for j in range(sample_size):
			s_id = sample_ids[j]
			adj_protein[i][j]  = all_neighbors[s_id][0]
			adj_inter[i][j]  = all_neighbors[s_id][1]


	return adj_protein,adj_inter,KG


def KG_construction(df):
	return

#get the latent representation of a pair of protein sequences
def embedding1():
	return

#one hot encode, the 20 amino acids are divided into 7 groups 
def OHE1():

	return

#transform a protein sequence into feature
def transfromToVector(seq):
	return

def readIdPair(fName=""):
	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()
	
	sFile = open(fName,'r')
	result = []

	couunt = 0
	while True:
		count += 1
		if count%1000 == 0:
			print("process {} pairs".format(count))		

		line = sFile.readline()
		if not line:
			break
		line = line.split(",")
		result.append([line[0],line[1]])

	sFile.close()
	return result

def readSeq(fName=""):
	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()
	sFile = open(fName,'r')
	result = dict()
	curS = ""
	curId = None
	idList = []

	line = sFile.readline()
	if line[0] != ">":
		print("incorrect format, each sequence should start with >")
		exit()
	curId = line.split("|")[1]

	count = 0
	while True:
		line = sFile.readline()
		if not line:
			break
		count += 1
		if line[0] == '>':
			idList.append(curId)
			curS = curS.replace("\n","")
			result[curId] = seq(_uId=curId,_sId=None,_seq=curS)
			curId = line.split("|")[1]
			curS = ""
		else:
			curS = curS + line
	print(idList[0:10])
	sFile.close()
	return result, idList

def encode_sequences(seqDict,idList):
	print("encode sequences")
	ECO = encoder("one_hot.txt")
	for i in idList:
		seqDict[i].vec = ECO.encode2(seqDict[i].seq)
	print(len(seqDict[idList[0]].vec))
	print(len(seqDict[idList[1]].vec))
	print(len(seqDict[idList[2]].vec))
	return seqDict

#if 'embeddings' not in sys.path:
#    sys.path.append('embeddings')

def process_help():
	printf("invalid command, this is the help page")
	return


if __name__ == "__main__":
	if len(sys.argv) < 2:
		exit()
	mode = sys.argv[1]
	fName = sys.argv[2]
	if mode == "-s": # standard mode, read file and id pair from files
		construct_KG(fName)
		#pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset),
        #        drug_vocab)  #save input of experimet
	elif mode =="-t":
		print("mode t")
