import sys
import os
import numpy as np
import tqdm #for print the progress note
import re
if 'FeatureComputaion' not in sys.path:
	sys.path.append('FeatureComputation')
import PSSM
import CojointTraid 
import PAAC
import CTDT
import ProtVect
import argparse
import sklearn
from collections import Counter
from gensim.models import Word2Vec
import pssmpro
import ProtBert
from sklearn.feature_selection import mutual_info_classif
# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, add
# from keras.layers.core import Flatten, Reshape
# from keras.layers.merge import Concatenate, concatenate, subtract, multiply
# from keras.layers.convolutional import Conv1D
# from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
# from keras.optimizers import Adam,  RMSprop

amino_list = ['A','G','V','I','L','F','P','Y','M','T','S','H','N','Q','W','R','K','D','E','C']

MAXLEN = 512

def check_files_exist(fList:list):
	for fName in fList:
		if not os.path.isfile(fName):
			print(f'error,input file {fName} does not exist')
			exit()
	return


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


class encoder():
	def __init__(self,eList): #embedding list 
		self.dic = {}
		self.dim = 0
		self.alphabet = []
		file = open(eList,'r')
		v = []
		while True:
			line = file.readline()	
			if not line:
				break
			t = line.split('\t')[0]
			self.alphabet.append(t)
			v = line.split('\t')[1]
			v = np.array([float(x) for x in v.split(' ')])
			self.dic[t] = v
		
		self.dim = len(v)
		#print("dimension of each {}".format(self.dim))
		for a in range(len(self.alphabet)):
			if self.alphabet[a] != amino_list[a]:
				print("the embedding list does not match: {} and {}.".format(self.alphabet[a],amino_list[a]))
				exit()

	def encode(self,seq):
		result = []
		if seq.find(' ') > 0:
			s = seq.strip().split()
		else:
			s = list(seq.strip())
		for s in seq:
			v = self.dic.get(s)
			if v is None:
				continue
			result.append(v)
		return result

	def encode2(self,seq,length=512):
		result = self.encode(seq)
		if len(result) > length:
			return result[:length]
		else:
			return np.concatenate((result,np.zeros((length-len(result),self.dim))))
		return

def selectFeature(matrix,pairList,relationList):
	n_component = 512
	ents = []

	for index in range(pairList.shape[1]):
		x = []
		for p in pairList:
			cur_id  = p[index]
			x.append(matrix[cur_id])
		x = np.array(x)
		ent = []
		for i in range(7):
			y = relationList[:,i]
			ent.append(mutual_info_classif(x,y) )
		ent = np.array(ent)
		ent = np.sqrt(ent)
		ent = np.sum(ent,axis=0)
		
		ents.append(ent)

	ents = np.array(ents)
	ents = np.sum(ents,axis=0)
	indices = np.argsort(ents)[::-1]
	indices = indices[0:512]
	f = open('FeatureComputation/indices3.txt','w')
	for i in indices:
		f.write(f'{i}\n')
	f.close()

	# print(ents[0:10])
	# print(indices[0:10])
	# print(ents[indices][0:10])
	# print(ents[indices][-10:])


	exit()
	return

#for the tsv file with the following format
#'item_id_a', 'item_id_b', 'mode', 'action','is_directional', 'a_is_acting', 'score'
def read_tsv(fName): 
	df = pd.read_table(fName,index_col=False)
	KG = nx.from_pandas_edgelist(df,"item_id_a","item_id_b",edge_attr=True)
	print(type(KG))
	print(list(nx.all_neighbors(KG,'9606.ENSP00000263025')))
	# KG2 = nx.graph
	# KG2.add_node(1)
	# KG2 = add_edge([1,2])

	return

def readSeq(seqPath=""):	
	idDict = dict()
	seqList = []
	pairList = []
	relationDir = {}
	relationList = []

	check_files_exist([seqPath])
	sFile = open(seqPath,'r')

	count = 0
	while True:
		line = sFile.readline().strip('\n')
		if not line:
			break
		tmp = re.split(',|\t',line)
		if tmp[0] not in idDict: #seq ID
			seqList.append(tmp[1])
			idDict[tmp[0]] = count
			count += 1
	for i,seq in enumerate(seqList):
		tmp = []
		for s in seq:
			if s not in amino_list:
				tmp.append(s)
		if len(tmp) > 0:
			for t in tmp:
				seqList[i] = seqList[i].replace(t,'')
	sFile.close()
	return seqList, idDict


#get the latent representation of a pair of protein sequences
def embedding1():
	return

#one hot encode, the 20 amino acids are divided into 7 groups 
def OHE1():

	return

#transform a protein sequence into feature
def transfromToVector(seq):
	return



def generatePSSM(seqFile:str,output:str,blast_db:str,blast_path='/home/user1/code/software/ncbi-install/bin/psiblast'):
	print(f'\nPSSM input {seqFile} {output} {blast_path} {blast_db}')

	number_of_cores = 8
	pssmpro.create_pssm_profile(seqFile,output,blast_path,blast_db,number_of_cores)
	return

#for the tsv file with the following format
#'item_id_a', 'item_id_b', 'mode', 'action','is_directional', 'a_is_acting', 'score'
def readIdPair(fName=""):


	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()
	
	sFile = open(fName,'r')
	sFile.readline()
	pair = []
	relation = []

	line = sFile.readline()
	line = line.split(",")
	pair.append([line[0],line[1]])

	couunt = 0
	while True:
		line = sFile.readline()
		if not line:
			break
		line = line.split(",")
		pair.append([line[0],line[1]])
		relation.append(line[2])
	sFile.close()
	return pair,relation

#read csv with the following format: 
#'item_id_a', 'item_id_b', 'mode', 'action','is_directional', 'a_is_acting', score, item_a_seq,item_b_seq



#fname with the following format: id1, id2, seq1, seq2
def readInteraction(fName=""):
#read seq with the following foramt in each line: id seq 
	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()
	sFile = open(fName,'r')
	idDict = dict() #index is id, value is numeric number
	seqList = [] #list of sequence
	pairList = []
	relationList = []
	count = 0
	seqName = []
	sFile.readline()

	class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}


	while True:
		line = sFile.readline()
		if not line:
			break
		tmp = line.strip('\n').split(',')
		if tmp[0] not in idDict:
			seqList.append(tmp[-2])
			idDict[tmp[0]] = count
			seqName.append(tmp[0])
			count += 1
		if tmp[1] not in idDict:
			seqList.append(tmp[-1])
			idDict[tmp[1]] = count
			seqName.append(tmp[1])
			count += 1
		pairList.append([tmp[0],tmp[1]])
		relationList.append(class_map[tmp[2]])

	for i,seq in enumerate(seqList):
		tmp = []
		for s in seq:
			if s not in amino_list:
				tmp.append(s)
		if len(tmp) > 0:
			for t in tmp:
				seqList[i] = seqList[i].replace(t,'')


	return pairList,np.asarray(relationList),idDict, seqList, seqName


#read seq that included in id dict
def readSepecificSeq(fName,idDict):
	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()
	sFile = open(fName,'r')
	seqList = [None for x in range(len(idDict))] #list of sequence

	count = 0
	while True:
		line = sFile.readline()
		if not line:
			break
		tmp = line.strip('\n').split('\t')
		if tmp[0] in idDict:
			seqList[idDict[tmp[0]]]=tmp[1]
	for seq in seqList:
		if seq is None:
			print("error, the sequence in pair list is not found")
			exit()
	return seqList

#
def CreateFasta(fName="",oDir=""):
	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()

	if not os.path.isdir(oDir):
		os.mkdir(oDir)

	sFile = open(fName,'r')	
	prefix = fName.split('/')[-1].split('.')[0]
	idDict = dict() #index is id, value is numeric number
	seqList = [] #list of sequence
	count = 0
	while True:
		line = sFile.readline()
		count += 1
		if not line:
			break
		oFile =open(oDir+f'/{prefix}{count}.fa','w')
		tmp = line.strip('\n').split('\t')
		oFile.write(f'>{tmp[0]}\n{tmp[1]}\n')
		oFile.close()
	sFile.close()
	return

def CalPos(seqList, **kw):
	result = []
	for seq in seqList:
		length = len(seq)
		tmp = (1+1+length)*length/2.0
		result.append(tmp/length)
	return np.array(result,dtype=float).reshape((len(seqList),-1))


def CalAAC(seqList, **kw):
	AA = kw['order'] if 'order' in kw else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	result = []
	for seq in seqList:
		tmp = []
		count = Counter(seq)
		for key in count:
			count[key] = count[key]/len(seq)
		for aa in AA:
			tmp.append(count[aa])
		result.append(tmp)
	return np.array(result,dtype=float)


def CalDPC(seqList:list, **kw):
	AA = kw['order'] if 'order' in kw else 'ACDEFGHIKLMNPQRSTVWY'
	result = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for seq in seqList:
		tmpCode = [0] * 400
		for j in range(len(seq) - 2 + 1):
			tmpCode[AADict[seq[j]] * 20 + AADict[seq[j+1]]] = tmpCode[AADict[seq[j]] * 20 + AADict[seq[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		result.append(tmpCode)
	return np.array(result,dtype=float)

# def readSeq2(fName:str): #read file from sequence that split into multiple lines
# 	if not os.path.isfile(fName):
# 		print("source file does not exisit, name {}\n".format(fName))
# 		exit()
# 	sFile = open(fName,'r')
# 	result = dict()
# 	curS = ""
# 	curId = None
# 	idList = []

# 	line = sFile.readline()
# 	if line[0] != ">":
# 		print("incorrect format, each sequence should start with >")
# 		exit()
# 	curId = line.split("|")[1]

# 	count = 0
# 	while True:
# 		line = sFile.readline()
# 		if not line:
# 			break
# 		count += 1
# 		if line[0] == '>':
# 			idList.append(curId)
# 			curS = curS.replace("\n","")
# 			result[curId] = seq(_uId=curId,_sId=None,_seq=curS)
# 			curId = line.split("|")[1]
# 			curS = ""
# 		else:
# 			curS = curS + line
# 	sFile.close()
# 	return result, idList


def encode_sequences(idDict, seqList):
	print("encode sequences")
	ECO = encoder("one_hot.txt")
	result = []
	for s in seqList:
		result.append(ECO.encode2(s))
	return np.array(result,dtype=float)

def seq_padding(seqList):
	paddedSeq = []
	seqNum = len(seqList)
	for i in range(seqNum):
		if len(seqList[i]) >= MAXLEN:
			paddedSeq.append(seqList[i][0:MAXLEN])
		else:
			paddedSeq.append(seqList[i])
			paddedSeq[i] += ' '*(MAXLEN-len(seqList[i]))

	return paddedSeq

def w2v(paddedSeq ,modelPath=None,size=20):
	model = Word2Vec.load(modelPath)

	result = []
	seqNum =  len(paddedSeq)
	size = len(model.wv[paddedSeq[0][0]])
	print(f'embedding size {size}')
	print(f'padding with maxLen {MAXLEN}')	

	for i in range(seqNum):
		tmp = []
		for j in range(MAXLEN):
			if paddedSeq[i][j] == ' ':
				tmp.extend(np.zeros(size))
			else:
				tmp.extend(model.wv[paddedSeq[i][j]])
		result.append(np.array(tmp))

	# scaler = sklearn.preprocessing.StandardScaler().fit(result)
	# result = scaler.transform(result)
	return np.array(result).reshape((seqNum,-1,size))

def word2type2(paddedSeq, modelPath='FeatureComputation/vec5_CTC.txt'):
	model = {}
	size = 13
	with open(modelPath,'r') as file:
		for line in file:
			content = re.split(' |\t',line)
			model[content[0]] = [float(c) for c in content[1:]]
			assert size == len(model[content[0]])
	result = []
	seqNum = len(paddedSeq)

	for i in range(seqNum):
		tmp = []
		for j in range(MAXLEN):
			if paddedSeq[i][j] == ' ':
				tmp.extend(np.zeros(size))
			else:
				tmp.extend(model[paddedSeq[i][j]])
		result.append(np.array(tmp))

	# scaler = sklearn.preprocessing.StandardScaler().fit(result)
	# result = scaler.transform(result)
	return np.array(result).reshape((seqNum,-1,size)) 


def word2type(paddedSeq, modelPath='FeatureComputation/amino-acid-type.txt'):
	model = {}
	size = 8
	with open(modelPath,'r') as file:
		for line in file:
			content = re.split(' |\t',line)
			model[content[0]] = [float(c) for c in content[1:]]
			assert size == len(model[content[0]])
	result = []
	seqNum = len(paddedSeq)

	for i in range(seqNum):
		tmp = []
		for j in range(MAXLEN):
			if paddedSeq[i][j] == ' ':
				tmp.extend(np.zeros(size))
			else:
				tmp.extend(model[paddedSeq[i][j]])
		result.append(np.array(tmp))

	# scaler = sklearn.preprocessing.StandardScaler().fit(result)
	# result = scaler.transform(result)
	return np.array(result).reshape((seqNum,-1,size)) 
#normalize 2d np array



def normalize_matrix(matrix:np.array):
	maxVals = matrix.max(axis=0)
	minVals = matrix.min(axis=0)

	for col in range(matrix.shape[1]):
		deno = maxVals[col] - minVals[col]
		if deno != 0.0:
			for row in range(matrix.shape[0]):
				matrix[row][col] = (matrix[row][col]-minVals[col])/deno
		else:
			for row in range(matrix.shape[0]):
				matrix[row][col] =  0.0	
	return matrix

def get_PSSM(idDict,PSSMPath=None,size=20):
	seqNum = len(idDict)
	fList = [s for s in range(seqNum)]
	for k in idDict.keys():
		fList[idDict[k]] = k
	for i in range(seqNum):
		fList[i] = PSSMPath+'/'+fList[i] +'.pssm'

	check_files_exist(fList)

	result = []
	for fName in fList:
		#print(f'PSSM file {fName}')
		file = open(fName,'r')
		for i in range(3):
			file.readline()	
		count = 0
		tmp = []
		while True:
			line = file.readline()
			if line == '\n' or count >= 512:
				break
			count +=1
			line = line.split()[2:22]
			scores = np.array([float(l) for l in line],dtype=float)
			tmp.extend(scores)
		if count < MAXLEN: #padding
			for i in range(count,MAXLEN):
				tmp.extend(np.zeros(20,dtype=float))  
		result.append(tmp)
	result = np.array(result,dtype=float)
	# scaler = sklearn.preprocessing.StandardScaler().fit(result)
	# result = scaler.transform(result)
	#'/home/user1/code/PPIKG/method/AFTGAN/string_both_PSSM/9606.ENSP00000000233.pssm'
	return result.reshape((seqNum,-1,size)) 

def sequences_embedding(seqList: list,idDict: dict,w2vPath=None,PSSMPath=None):
	header = ""
	#two optional ways for embedding, hot encoding and word2Vec
	#the w2v model is from DeepFE-PPI
	paddedSeq = seq_padding(seqList)
	print(f'word2vec path {w2vPath}')


	eMatrix = w2v(paddedSeq,w2vPath)
	eMatrix2 = word2type(paddedSeq)
	eMatrix = np.concatenate((eMatrix,eMatrix2),axis=-1)
	#eMatrix3, fMatrix2 = ProtBert.word_embedding(paddedSeq)


	#eMatrix = np.concatenate((eMatrix,eMatrix3),axis=-1)
	print(f'shape of eMatrix {eMatrix.shape}')
	if PSSMPath:
		print(f'PSSM path: {PSSMPath}')
		eMatrix3=get_PSSM(idDict,PSSMPath)
		eMatrix = np.concatenate((eMatrix,eMatrix3),axis=-1)
	
	eMatrix = eMatrix.reshape((len(seqList),-1))
	#scaler = sklearn.preprocessing.StandardScaler().fit(eMatrix)
	#eMatrix = scaler.transform(eMatrix)
	eMatrix = sklearn.preprocessing.normalize(eMatrix,axis=0)
	
	eMatrix = eMatrix.reshape((len(seqList),MAXLEN,-1))
	
	curShape = 0
	col_name = []
	fMatrix = CojointTraid.CalCJ(seqList); header+="Cojoint Traid;" #dimension 343, alternative: sum the vector, wait to be tested
	#fMatrix = CalDPC(seqList)	#dimension 400

	#fMatrix = np.concatenate((fMatrix,CalDPC(seqList)),axis=1) # dimension 400
	fMatrix = np.concatenate((fMatrix,CalAAC(seqList)),axis=1)  #dimension 20
	fMatrix = np.concatenate((fMatrix,PAAC.CalPAAC(seqList)),axis=1)  #dimension 50
	fMatrix = np.concatenate((fMatrix,CTDT.CalCTDT(seqList)),axis=1) #dimension 39
	fMatrix = np.concatenate((fMatrix,ProtVect.CalProtVec(seqList)),axis=1) #dimension 1
	
	fMatrix = np.concatenate((fMatrix,CalPos(seqList)),axis=1)  #dimension 1
	
	#fMatrix = np.concatenate((fMatrix,fMatrix2),axis=1) 
	a = 343+400+20+50+39+1+1
	print(f'\n the shape of eMatrix: {eMatrix.shape} ,fMatrix: {fMatrix.shape} \n')

	# if PSSMPath:
	# 	fMatrix = np.concatenate((fMatrix, PSSM.CalPSSM(prefix=PSSMPath, fNum=len(seqList))),axis=1) #dimesnion 20
	# 	print(PSSM.CalPSSM(prefix=PSSMPath, fNum=len(seqList)).shape)
	# 	exit()
	#/mnt/data6t/EC/PPI/data/uniprot/uniref50.fasta
	#alternatinve: dimensionality reduction
	fMatrix=sklearn.preprocessing.normalize(fMatrix,axis=0)
	#scaler2 = sklearn.preprocessing.StandardScaler().fit(fMatrix)
	#fMatrix = scaler2.transform(fMatrix)

	#backup, for dimension reduction


	return fMatrix,col_name, eMatrix


