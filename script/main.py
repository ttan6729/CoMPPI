import sys
import os
import argparse
import torch
import keras
import pandas as pd
import Model
from Model import CoMPPI
from Model import ModelConfig
import embedding as EB
import numpy as np
import random
import kg

from sklearn.model_selection import train_test_split,StratifiedKFold

MAX_SCORE = 0.0
DATA_INDEX = -1 

#torch keras pandas numpy tensorflow
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_random_string(length):
	# choose from all lowercase letter
	letters = string.ascii_lowercase
	result_str = ''.join(random.choice(letters) for i in range(length))
	return result_str

def str2bool(v):
	"""
	Converts string to bool type; enables command line 
	arguments in the format of '--arg1 true --arg2 false'
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	return

def sort_and_round(data,digits=4):
	data.sort(reverse=True)
	result = [round(s,digits) for s in data]
	return result
def round_list(data,digits=4):
	result = []
	for i in data:	
		result.append(round(i,digits))
	return result



def train_and_evaluate(model,args,train_d,test_d,dataIndex=-1,if_save=False):
	np.random.shuffle(train_d)
	depth = args.d
	batch_size = args.b #default set as 256
	warmup_epoch = 10
	base_lr = 3.5e-04
	epoch_unit = 1
	scores = []
	sample_count = train_d.shape[0]
	total_steps = int(args.e * sample_count / batch_size)
	warmup_steps = int(warmup_epoch * sample_count / batch_size)
	warmup_batches = warmup_epoch * sample_count / batch_size
	warm_up_lr = Model.WarmUpCosineDecayScheduler(learning_rate_base=base_lr,
									total_steps=total_steps,warmup_learning_rate=4e-06,
									warmup_steps=warmup_steps,hold_base_rate_steps=5)
	y_pred = None
	for e in range(0,args.e+1,epoch_unit):
		np.random.shuffle(train_d)
		sv = f'{args.sv}'
		checkpoint=None
		train_history = model.fit(x=[train_d[:,0:1],train_d[:,1:2]],y=train_d[:,2:],batch_size=batch_size, epochs=epoch_unit,callbacks=[warm_up_lr])
		score, y_tmp = model.predict_and_eval(x=[test_d[:,0:1],test_d[:,1:2]], y=test_d[:,2:],output=args.o,epoch = e)		
		scores.append(score)
		if score == max(scores):
			y_pred = y_tmp

		global MAX_SCORE
		if score > MAX_SCORE:
			MAX_SCORE = score
			if if_save:
				model.save_weights(args.sv)

				global DATA_INDEX
				if DATA_INDEX != dataIndex:
					DATA_INDEX = dataIndex
					model.record(oPath=args.sv+'.data',test_set=test_d,seqPath=args.i1,relPath=args.i2,
						modelPath = args.sv)
	scores.sort(reverse=True)
	with open(args.o,'a') as f:
		f.write('\n')		

	return scores, y_pred, test_d[:,2:]

 
def run(seqPath,relPath,args):
	seqList, idDict = EB.readSeq(seqPath)	
	KG = kg.KnowledgeGraph(relPath,seqPath,idDict,sample_size =3,depth=3)
	pairList, relationList, invDict = KG.pairList, KG.relationList, KG.invDict
	w2vPath = 'FeatureComputation/word2vec/wv_swissProt_size_20_window_16.model'
	fMatrix, col_name,eMatrix = EB.sequences_embedding(seqList,idDict,w2vPath=w2vPath,PSSMPath=args.PSSM) 
	gMatrix = KG.generate_feature(fMatrix) #the matrix after graph convolution
	relationNum = relationList.shape[1]
	dataset = np.concatenate((pairList,relationList),axis=1)

	config = ModelConfig(lr=args.lr,beta1 = args.beta1)
	if args.o:
		file = open(args.o,'w')
		command = ' '.join(arg for arg in sys.argv)
		file.write(f'command: {command}\n')
		file.write(f'mode: {args.m}\n')
		file.write(f'{config.get_parameter()}')
		file.write(f'filePath: {seqPath} {relPath}\n')
		file.write(f'feature shape: {fMatrix.shape}, embedding shape: {eMatrix.shape}\n')
		file.write(f'comment: {args.co}\n')
		file.write(f'epoch: {args.e}\n')
		file.write(f'mulity layer and concatenate\n')
		if args.s:
			file.write(f'save path: {args.sv}\n')
		file.close()

	scores = []
	scores_each_itr = []
	max_scores = []
	actual = []
	pred = []
	if args.m == 's1': 
		K_fold = 5
		subsets = dict()
		n_sample = dataset.shape[0]
		n_subsets = int(n_sample/K_fold)
		remain=set(range(0,n_sample))
		
		for i in reversed(range(0,K_fold-1)):
			subsets[i]=random.sample(list(remain),n_subsets)
			remain=remain.difference(subsets[i]) 
			subsets[K_fold-1]=remain

		for i in reversed(range(0,K_fold)):
			test_data=dataset[list(subsets[i])]
			print(f'shape of test set: {test_data.shape}')	
			train_d = []
			test_d = test_data
			test_d = test_data
			for j in range(0,K_fold):
				if i != j:
					train_d.extend(dataset[list(subsets[j])])
			with open(args.o,'a') as my_file:
				my_file.write('\n')
			train_d = np.array(train_d)	
			comppi =CoMPPI(config,idDict,gMatrix,eMatrix,seqList=seqList,labelNum=relationNum,eShape=eMatrix.shape
			,depth=3,invDict=invDict,ablation=args.ab)
			comppi.record(oPath=args.sv+'_'+str(i)+'.data',test_set=test_d,seqPath=args.i1,relPath=args.i2)			
			tmp_scores, y_pred, y_actual = train_and_evaluate(comppi,args,train_d,test_d,dataIndex=i,if_save=args.s)
			
			pred.extend(y_pred.tolist())
			actual.extend(y_actual.tolist())
			scores.extend(tmp_scores) 
			scores_each_itr.append(tmp_scores)
		

	else:
		train_indice, test_indices = [], []
		if args.m == 's2':
			KG.read_test_set(args.i3)
			train_indices = KG.construct_training_set()
			test_indices = KG.test_set
		for i in range(args.itr):
			if args.m == 's3': #breath first search
				train_indices ,test_indices = KG.split_dataset_bfs(test_percentage=0.2)
			elif args.m == 's4':
				train_indices ,test_indices = KG.split_dataset_dfs(test_percentage=0.2)
			
			comppi =CoMPPI(config,idDict,gMatrix,eMatrix,seqList=seqList,labelNum=relationNum,eShape=eMatrix.shape
			,depth=3,invDict=invDict,ablation=args.ab)
			train_d,test_d = dataset[train_indices,:], dataset[test_indices,:]
			if args.mp:
				comppi.model.load_weights(args.mp)
				score = comppi.predict_and_eval(x=[test_d[:,0:1],test_d[:,1:2]], y=test_d[:,2:],output=args.o)	
				print(f'score is {score}')
				break

			tmp_scores,y_pred, y_actual = train_and_evaluate(comppi,args,train_d,test_d,dataIndex=i,if_save=args.s)
			scores.extend(tmp_scores)
			scores_each_itr.append(tmp_scores)
			
			global MAX_SCORE
			if tmp_scores[0] == MAX_SCORE:
				pred = y_pred.tolist()
				actual = y_actual.tolist()

	result = {'pred':torch.tensor(pred),'actual':torch.tensor(actual)}
	torch.save(result,args.rp)
	scores = sort_and_round(scores,3)
	max_scores = []
	for s in scores_each_itr:
		max_scores.append(sort_and_round(s)[0])

	with open(args.o,'r+') as my_file:
		file_data = my_file.read()
		my_file.seek(0,0)
		line = f'average microF1 {round(sum(scores)/len(scores),5)}\n'		
		line += f'highest score: {round(MAX_SCORE,5)}, index:{DATA_INDEX}\n'
		line += f'average of highest score each itr:{round(sum(max_scores) / len(max_scores),5)},{round_list(max_scores)}\n'
		tmp = args.sv+'.data'
		line += f'recorded dataset path: {tmp}\n'
		line += f'recorded prediction path: {args.rp}\n'
		line += 'highest score of each itr\n'
		for cur_scores in scores_each_itr:
			line += f'{sort_and_round(cur_scores,3)[0:5]}\n'

		my_file.write(line + '\n' + file_data)

	return

def get_args_parser():
	parser = argparse.ArgumentParser('PPIM',add_help=False)
	parser.add_argument('-m',default='s',type=str,help='mode')
	parser.add_argument('-o',default='output.txt',type=str,help='output path, the suffix will be used as path for saveing model and data')
	parser.add_argument('-sf', default=None,type=str,help='optional input, contains path for sequence and relation file')
	parser.add_argument('-i1',default=None,type=str,help='sequence file')
	parser.add_argument('-i2',default=None,type=str,help='relation file')
	parser.add_argument('-i3',default=None,type=str,help='file path of test set indices (for mode s4)')
	parser.add_argument('-e',default=100,type=int,help='epochs')
	parser.add_argument('-b', default=256, type=int,help='batch size')
	parser.add_argument('-s', default=False, type=str2bool,help='save the best mode')
	parser.add_argument('-PSSM',default=None,type=str,help='path of PSSM')
	parser.add_argument('-blastdb',default=None,type=str,help='path of blast db')
	parser.add_argument('-d',default=3,type=int,help='depth') 
	parser.add_argument('-sv',default=None,type=str,help='model save path')
	parser.add_argument('-itr',default=3,type=int,help='iteration number for validation')
	parser.add_argument('-co',default='default comment',type=str,help='comment')
	parser.add_argument('-dp',default=False,type=str2bool,help='if add defaultPath path')
	parser.add_argument('-lr',default=5e-04,type=float,help='learning rate')
	parser.add_argument('-beta1',default=0.9,type=float,help='beta1 for adad optimizer')
	parser.add_argument('-dm',default=None,type=int,help='dataset mode') 
	parser.add_argument('-mp',default=None,type=str,help='the path of the model weight to be loaded')
	parser.add_argument('-tp',default=None,type=str,help='the path of the test set that to be saved')
	parser.add_argument('-rp',default=None,type=str,help='the result path for file')
	parser.add_argument('-ab',default=0,type=int,help='ablation study') 
	return parser




if __name__ == "__main__":
	parser = argparse.ArgumentParser('PPIM', parents=[get_args_parser()],add_help=True)
	args = parser.parse_args()

	if args.dp:
		print('add default path')
		defaultPath = '/home/user1/code/PPIKG/method/AFTGAN/data/'
		args.i1 = defaultPath + args.i1
		args.i2 = defaultPath + args.i2

	if args.sf:
		with open(args.sf,'r') as f:
			args.i1 = f.readline().strip()
			args.i2 = f.readline().strip()

	if args.sv:
		args.sv = args.o.split('.')[0]
		args.rp = args.sv+'.pt'
		os.system(f'rm -f {args.rp}')

	# if os.path.exists(args.o) and args.o != 'output.txt':
	# 	print(f'output file {args.o} already exist')
	# 	exit()

	print(f'path of sequence file {args.i1}')
	print(f'path of relation file {args.i2}')

	if args.m[0] == 's' : #s1: cross validation, s2: read, s3 bfs, s4 dfs
		seqPath = args.i1
		relPath = args.i2
		run(seqPath,relPath,args)

	elif args.m == 'gs': 
		print('generate PSSM with given dataset')
		PSSMPath = args.PSSM
		seqPath = args.i1
		EB.generatePSSM(seqPath,PSSMPath,args.blastdb,'/home/user1/code/software/ncbi-install/bin/psiblast')
		
	elif args.m == 'sp':
		print('split dataset')
		seqPath = args.i1
		relPath = args.i2
		seqList, idDict = EB.readSeq(seqPath)	
		KG = kg.KnowledgeGraph(relPath,idDict,sample_size =3,depth=3,is_split=True)

	elif args.m =='r':
		print('read dataset from ')
		dataPath = args.i1