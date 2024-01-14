import sys
import os
import numpy as np
import argparse
import re
from collections import defaultdict
from collections import Counter
import random
def sorted_pair(id1,id2):
	if id1<id2:
		return [id1,id2]
	return [id2,id1]

#generate id based on pair of numeric values
def generate_id(id1,id2):
	if id1<id2:
		return str(id1)+'_'+str(id2)
	return str(id2)+'_'+str(id1)

class KnowledgeGraph:
	def __init__(self,relPath,seqPath,idDict,sample_size =5,depth=3,is_split=False):
		print(f'construct knowledge graph, sample size is {sample_size}')
		class_dir = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}
		self.class_num = len(class_dir)
		self.sample_size = sample_size
		self.relPath = relPath
		self.seqPath = seqPath
		node_num = len(idDict)
		self.adj_node = np.zeros(shape=(self.class_num,node_num,sample_size),dtype=int)
		self.depth = depth
		self.test_test = []
		rFile = open(relPath,'r')
		header = rFile.readline() #assume the first three are id of seq A, id of seq B, mode
		self.relationDir = {}  #key: pair that respresented by string 
		relationList = [] #the list of intercation type
		pairList = [] # the list of id of protein in each pair
		uniqueCount = 0
		dupRelation = 0
		while True: 
			line = rFile.readline().strip('\n')
			if not line:
				break
			tmp = re.split(',|\t',line)
			id1,id2,mode, is_dir = idDict[tmp[0]], idDict[tmp[1]], class_dir[tmp[2]], tmp[4]
			newPair = sorted_pair(id1,id2)
			newId = str(newPair[0]) +'_'+ str(newPair[1]) 
		
			if newId not in self.relationDir.keys():
				self.relationDir[newId] = uniqueCount
				uniqueCount += 1
				tmp = []
				for i in range(self.class_num):
					tmp.append(0)
				relationList.append(tmp)
				pairList.append(newPair)
			relationList[ self.relationDir[newId] ][mode] = 1 
		rFile.close()

		KGs = []
		for i in range(self.class_num):
			KGs.append(defaultdict(list))

		print(f'num of unique pair {uniqueCount}')

		for i in range(len(pairList)):
			id1, id2 = pairList[i][0], pairList[i][1]
			for j in range(self.class_num):
				if relationList[i][j] == 1:
					KGs[j][id1].append(id2)
					KGs[j][id2].append(id1)

		for j in range(self.class_num):
			for i in range(len(idDict)):
				if len(KGs[j][i]) == 0:
					KGs[j][i].append(-1)


		#assert len(KGs) == self.class+num
		for i in range(self.class_num):
			for j in range(node_num):
				all_neighbors = KGs[i][j]
				neigh_num = len(all_neighbors)
				#uniformly sample
				sample_indices = np.random.choice(neigh_num,sample_size,
					replace=False if neigh_num >= sample_size else True)
				self.adj_node[i][j] = np.array([all_neighbors[index] for index in sample_indices])
	
		self.idDict = idDict
		self.invDict = {v: k for k, v in self.idDict.items()}
		self.class_dir = class_dir
		self.relationList = np.array(relationList,dtype=int)
		self.pairList = np.array(pairList,dtype=int)

		return 

	#feature for GNN module
	def generate_feature(self,fMatrix):
		assert self.adj_node.shape[1] == fMatrix.shape[0]
		nodeNum = fMatrix.shape[0]
		featureNum = fMatrix.shape[1]
		result = np.zeros((nodeNum,self.class_num,self.depth,featureNum))
		for i in range(nodeNum):
			for j in range(self.class_num):
				tmpId = [i]
				for k in range(self.depth):
					tmpArray = np.zeros((self.sample_size**k,featureNum),dtype=float)
					if k != 0:
						tmpId = np.array([self.adj_node[j][t] for t in tmpId]).ravel()
					for index,t in enumerate(tmpId):
						if t != -1:
							tmpArray[index] = fMatrix[t]
					result[i][j][k] = np.mean(tmpArray,axis=0)	
		return result	

	def construct_training_set(self,test_indices=None):
		if not test_indices and len(self.test_set)==0:
			raise Exception("error in constructing indicies for test set: the original test set is not found")
		if not test_indices:
			test_indices = self.test_set
		all_indices = [i for i in range(len(self.pairList))]
		training_indices = list( set(all_indices).difference( set(test_indices) ) )
		assert len(self.pairList) == (len(training_indices)+len(test_indices)), "error, the size of training and test set doesn't match"
		return training_indices

	def read_test_set(self,save_path):
		file = open(save_path,'r')
		self.test_set =[]
		seqPath = file.readline()[1:]
		relPath = file.readline()[1:]
		print(f'pair list source: {relPath}, test set source: {save_path}')
		while True:
			newLine = file.readline().strip('\n')
			if not newLine:
				break
			newPair = re.split(' |\t',newLine)
			id1,id2 = self.idDict[newPair[0]],self.idDict[newPair[1]]
			newId = generate_id(id1,id2) #str(newPair[0]) +'_'+ str(newPair[1]) 
			# if newId not in self.relationDir.keys():
			# 	raise Exception(f'failed to load index from {save_path}')
			self.test_set.append( self.relationDir[newId] ) 
		self.test_set = np.array(self.test_set,dtype=int)
		return

	def record(self,oName='test.txt',test_set=None): #record the sequence name of each pair in re
		oFile = open(oName,'w')
		oFile.write(f'>{self.seqPath}\n')
		oFile.write(f'>{self.relPath}\n')
		
		if not test_set:
			test_set = self.test_set

		for i in test_set:
			pair = self.pairList[i]
			name1 = self.invDict[pair[0]]
			name2 = self.invDict[pair[1]]
			pair = sorted_pair(name1,name2)
			oFile.write(f'{pair[0]}\t{pair[1]}\n')
		oFile.close()

		return


#use breadth search to get a subgraph which represents test set,
#in each edge, the id should in ascending order
	def split_dataset_bfs(self,node_to_edge_index=None, test_percentage=0.1,src_path=None): #list of interaction, percentage of
		# for i in range(10):
		# 	print(random.randint(0,10))
		# exit()
		if not node_to_edge_index:
			node_to_edge_index = defaultdict(list)
			for i in range(len(self.pairList)):
				id1, id2 = self.pairList[i][0], self.pairList[i][1]
				node_to_edge_index[id1].append(i)
				node_to_edge_index[id2].append(i)

		node_num = len( node_to_edge_index.keys() )
		test_size = int(len(self.pairList) * test_percentage)
		self.test_set = []
		queue = []
		visited = []

		random_index = random.randint(0, node_num-1)
		while len( node_to_edge_index[random_index] ) > 5:
			random_index = random.randint(0, node_num-1)
		queue.append(random_index)
		print(f'root level {len( node_to_edge_index[random_index])}')
		count = 0
		#print(node_to_edge_index[random_index])

		while len(self.test_set) < test_size:
			if len(queue) == 0:
				print('bfs split meet root level 0')
				exit()
			cur_node = queue.pop(0) 
			visited.append(cur_node)
			for edge_index in node_to_edge_index[cur_node]:
				if edge_index not in self.test_set:
					self.test_set.append(edge_index)
					id1,id2 = self.pairList[edge_index][0],self.pairList[edge_index][1]
					next_node = id1
					if id1 == cur_node:
						next_node = id2
					if next_node not in visited and next_node not in queue:
						queue.append(next_node)
				else:
					continue

		self.test_set = np.array(self.test_set,dtype=int)
		training_set = self.construct_training_set()

		return training_set,self.test_set



	def split_dataset_dfs(self,node_to_edge_index=None, test_percentage=0.2,src_path=None):
		if not node_to_edge_index:
			node_to_edge_index = defaultdict(list)
			for i in range(len(self.pairList)):
				id1, id2 = self.pairList[i][0], self.pairList[i][1]
				node_to_edge_index[id1].append(i)
				node_to_edge_index[id2].append(i)

		node_num = len( node_to_edge_index.keys() )
		test_size = int(len(self.pairList) * test_percentage)
		self.test_set = []
		stack = []
		visited = []

		random_index = random.randint(0, node_num-1)
		while len( node_to_edge_index[random_index] ) > 5:
		 	random_index = np.random.randint(0, node_num-1)
		print(f'random index {random_index},root level {len( node_to_edge_index[random_index])}')

		stack.append(random_index)

		while(len(self.test_set) < test_size):
			if len(stack) == 0:
				print('dfs split meet root level 0')
				exit()
			cur_node = stack[-1]
			if cur_node in visited:
				flag = True
				for edge_index in node_to_edge_index[cur_node]:
					if flag:
						id1,id2 = self.pairList[edge_index][0],self.pairList[edge_index][1]
						next_node = id1 if id2 == cur_node else id2
						if next_node in visited:
							continue
						else:
							stack.append(next_node)
							flag = False
					else:
						break
				if flag:
					stack.pop()
				continue
			else:
				visited.append(cur_node)
				for edge_index in node_to_edge_index[cur_node]:
					if edge_index not in self.test_set:
						self.test_set.append(edge_index)
		self.test_set = np.array(self.test_set,dtype=int)
		return self.construct_training_set(),self.test_set

