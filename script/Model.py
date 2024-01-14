import sys
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from gensim.models import Word2Vec
from keras.layers import Layer #from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import *
from termcolor import colored
from keras.regularizers import l2
import sklearn
import tensorflow_addons as tfa
import math
import kg
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to disable warning
class ModelConfig(object):
	def __init__(self,lr=5e-04,beta1=0.9):
		self.sample_size = 4 #neighbor sampling size
		self.n_depth = 3
		self.optimizer = keras.optimizers.Adam(learning_rate=1e-03) # learning_rate=0.001,beta_1=0.9,beta_2=0.999, epsilon=1e-07,
		self.lr = lr
		self.beta_1 = beta1
		self.beta_2 = 0.999
		self.epsilon = 1e-07
		self.l2_weight = 1e-7

	def get_parameter(self):
		result = f'optimizer: lr:{self.lr}, beta1:{self.beta_1}, beta2: {self.beta_2},  epsilon:{self.epsilon}\n'
		return result


#graph matrix shape: [sample_num,class_num,depth,featureNum]
class CoMPPI(object):
	def __init__(self,config:ModelConfig,idDict: dict, gMatrix:np.array
		,eMatrix:np.array,seqList=None,labelNum=0, window_size=20, eShape=[],depth=3,invDict=None,ablation=0): 
		self.idDict = idDict
		self.seqList = seqList
		self.config = config
		self.optimizer = keras.optimizers.Adam(learning_rate=config.lr,beta_1=config.beta_1,
			beta_2=config.beta_2,epsilon=config.epsilon)

		self.featureNum = gMatrix.shape[3]
		self.gMatrix = K.variable(gMatrix,name='gMatrix',dtype='float32')
		self.eMatrix = K.variable(eMatrix,name='eMatrix',dtype='float32')		
		self.labelNum = labelNum
		self.sampleNum = eMatrix.shape[0]
		self.window_size = window_size
		self.embeddingShape = eShape
		self.depth = depth
		self.invDict = invDict
		self.model = self.build(ablation)
		return

	def build(self,ablation=0):
		if ablation ==1:
			return self.build_ablation1()
		elif ablation ==2:
			return self.build_ablation2()

		model_input =[Input(shape=(1,),name='protA'),Input(shape=(1,),name='protB')]  	
		print(colored("begin build model",'red'))

		inputA = model_input[0]
		inputB = model_input[1]
		graphA = self.graphConv(inputA,'graph_A')
		graphB = self.graphConv(inputB,'graph_B')

		featureA1 = Lambda(lambda x: K.gather(self.eMatrix,K.cast(x,dtype='int32')),
			name='featureA1')(inputA)
		featureB1 = Lambda(lambda x: K.gather(self.eMatrix,K.cast(x,dtype='int32')),
			name='featureB1')(inputB)

		textA1 = self.textCNN(featureA1,'textCNN_A1')
		textB1 = self.textCNN(featureB1,'textCNN_B1')

		o1 = Concatenate(axis=-1)([graphA,textA1])
		o2 = Concatenate(axis=-1)([graphB,textB1])

		mul = tf.keras.layers.Multiply()([o1,o2])
		output = Concatenate(axis=-1)([mul,o1,o2])

		scores = Dense(self.labelNum,activation='sigmoid')(output)
		metrics = [tf.keras.metrics.BinaryAccuracy(),tfa.metrics.F1Score(num_classes=self.labelNum, 
					  average='micro',threshold=0.5)]
		model = keras.models.Model(model_input,scores,name='a')
		model.compile(optimizer=self.optimizer,loss='binary_crossentropy',metrics=metrics)
		print(colored('finish builing model','red'))

		return model

	def save(self,fp,save_format):
		self.model.save(fp,save_format=save_format)
		return

	def save_weights(self,fp):
		self.model.save_weights(fp+'.h5')
		return

	def load(self,fp=''):
		self.model.load_weights(fp+'.h5')

	def graphConv(self,inputId,prefix=''):
		get_neighbor = Lambda(lambda x: self.get_neighbour_featrue_by_id(x),name=prefix+'get_neighbor')
		get_feature = Lambda(lambda x: self.get_featrue_by_depth(x),name=prefix+'get_feature')
		output = []
		for i in range(self.labelNum):
			features = get_neighbor([i,inputId]) #neighbour of the ith knowledge graph
			current_label_feature = get_feature([features,0])
			for j in range(1,self.depth):
				current_level = get_feature([features,j])
				current_label_feature = ConcatAggregator(activation='tanh' if j == self.depth-1 else 'relu',
					regularizer=l2(1e-7))([current_label_feature,current_level])	
				current_label_feature = BatchNormalization(epsilon=1e-6)(current_label_feature)
			output.append(current_label_feature)
		output = Concatenate(axis=-1)(output)
		output = Lambda(lambda x: K.squeeze(x,axis=1))(output)
		return output

	def textCNN(self,feature,prefix='',drop_rate=0.3):
		feature = Lambda(lambda x: K.squeeze(x,axis=1))(feature)
		seqLen = feature.shape[1]
		k_sizes = [11,13,15]
		outputs = []
		filter_Num = 256
		
		for i, k_size in enumerate(k_sizes):
			newLayer = Conv1D(filters=filter_Num,kernel_size=k_size,strides=1,padding='valid',
				kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
				bias_initializer=keras.initializers.constant(value=0.1))(feature) 
			#newLayer = BatchNormalization(epsilon=1e-6)(newLayer)
			newLayer = Activation(keras.activations.relu)(newLayer) 
			newLayer = MaxPooling1D(pool_size=seqLen-k_size+1,strides=1,padding='valid')(newLayer)
			outputs.append(newLayer)
		outputs = Concatenate(axis=-1)(outputs)
		outputs = Flatten()(outputs)
		outputs = Dropout(drop_rate)(outputs)
		return outputs

	def record(self,oPath,test_set,seqPath,relPath,modelPath=None):
		oFile = open(oPath,'w')
		oFile.write(f'>{seqPath}\n')
		oFile.write(f'>{relPath}\n')

		for pair in test_set:
			name1 = self.invDict[pair[0]]
			name2 = self.invDict[pair[1]]
			pair = kg.sorted_pair(name1,name2)
			oFile.write(f'{name1}\t{name2}\n')
		oFile.close()
		return


	def fit(self,x,y,batch_size=10,epochs=5,val_data=None,callbacks=None):		
		history = self.model.fit(x=x,y=y,batch_size=batch_size,epochs=epochs,validation_data=val_data,callbacks=callbacks)
		#print(history.history.keys())
		if val_data:
			print('validation loss:',end =" ")
			for vl in history.history['val_loss']:
				print('%0.3f'%vl,end =" ")
			print('')
		return history
 
	def score(self,y,y_pred,output=None,model='ensemble model'): 
		TP = np.zeros((self.labelNum),dtype=float)
		FP = np.zeros((self.labelNum),dtype=float)
		FN = np.zeros((self.labelNum),dtype=float)
		TN = np.zeros((self.labelNum),dtype=float)
		for labelId in range(self.labelNum):
			 for i in range(y.shape[0]):
				 if y[i][labelId] == 1 and y_pred[i][labelId] == 1:
					 TP[labelId] += 1
				 elif y[i][labelId] == 0 and y_pred[i][labelId] == 0:
					 TN[labelId] += 1    
				 elif y[i][labelId] ==  1 and y_pred[i][labelId] == 0:
					 FN[labelId] += 1 
				 elif y[i][labelId] == 0 and y_pred[i][labelId] == 1:
					 FP[labelId] += 1 
		accuracy = (np.sum(TP)+np.sum(TN))/(np.sum(TP)+np.sum(TN)+np.sum(FP)+np.sum(FN))
 
		microF1 = np.sum(TP)/(np.sum(TP)+0.5*np.sum(FN)+0.5*np.sum(FP))
		loss = sklearn.metrics.hamming_loss(y,y_pred)
		precision =  np.sum(TP)/(np.sum(TP)+np.sum(FP))
		recall =  np.sum(TP)/(np.sum(TP)+np.sum(FN))
		#model: {model}
		record = f'sample#:{y.shape[0]},acc {accuracy:.4f},microF1 {microF1:.4f},precision {precision:.2f},recall {recall:.2f},loss {loss:.2f}, f1 of each class:'
		f1s = []
		for i in range(self.labelNum):
			p = TP[i]/(TP[i]+FP[i])
			r = TP[i]/(TP[i]+FN[i])
			f1s.append(round( 2*(p*r)/(p+r+0.001), 3 ))
			record += f',{f1s[i]}'
		
		print(record)
		return record,microF1

	def predict_and_eval(self,x,y,threshold=0.5,output=None,epoch=0):
		y_pred = self.model.predict(x)
		result = np.copy(y_pred)
		for i in range(y_pred.shape[0]):
			for j in range(y_pred.shape[1]):
				y_pred[i][j] = 1 if y_pred[i][j] > 0.5 else 0

		y_pred = y_pred.astype(int)
		record, microF1 = self.score(y,y_pred)
		record = f'e:{epoch},{record}\n'
		if output:
			file = open(output,'a')
			file.write(record)
			file.close()
		return microF1,result

	def load_weight(self,path):
		self.model.load_weights(path)
		return

	def get_model(self):
		return self.model

	def build_ablation1(self):
		model_input =[Input(shape=(1,),name='protA'),Input(shape=(1,),name='protB')]  	
		inputA = model_input[0]
		inputB = model_input[1]
		graphA = self.graphConv(inputA,'graph_A')
		graphB = self.graphConv(inputB,'graph_B')

		mul = tf.keras.layers.Multiply()([graphA,graphB])
		output = Concatenate(axis=-1)([mul,graphA,graphB])

		scores = Dense(self.labelNum,activation='sigmoid')(output)
		metrics = [tf.keras.metrics.BinaryAccuracy(),tfa.metrics.F1Score(num_classes=self.labelNum, 
					  average='micro',threshold=0.5)]
		model = keras.models.Model(model_input,scores,name='a')
		model.compile(optimizer=self.optimizer,loss='binary_crossentropy',metrics=metrics)
		print(colored('finish builing model, abltaion1','red'))

		return model

	def build_ablation2(self):
		model_input =[Input(shape=(1,),name='protA'),Input(shape=(1,),name='protB')]  	
		print(colored("begin build model",'red'))

		inputA = model_input[0]
		inputB = model_input[1]

		featureA1 = Lambda(lambda x: K.gather(self.eMatrix,K.cast(x,dtype='int32')),
			name='featureA1')(inputA)
		featureB1 = Lambda(lambda x: K.gather(self.eMatrix,K.cast(x,dtype='int32')),
			name='featureB1')(inputB)

		textA1 = self.textCNN(featureA1,'textCNN_A1')
		textB1 = self.textCNN(featureB1,'textCNN_B1')

		mul = tf.keras.layers.Multiply()([textA1,textB1])
		output = Concatenate(axis=-1)([mul,textA1,textB1])

		scores = Dense(self.labelNum,activation='sigmoid')(output)
		metrics = [tf.keras.metrics.BinaryAccuracy(),tfa.metrics.F1Score(num_classes=self.labelNum, 
					  average='micro',threshold=0.5)]
		model = keras.models.Model(model_input,scores,name='a')
		model.compile(optimizer=self.optimizer,loss='binary_crossentropy',metrics=metrics)
		print(colored('finish builing model, abltaion2','red'))

		return model

	#only return ID
	def get_receptive_field(self,entity):
		result = [entity]
		n_sample = K.shape(self.adj_node)[1]
		for i in range(self.config.n_depth):
			new_nodes = K.gather(self.adj_node,K.cast(result[-1],dtype='int32'))
			result.append(K.reshape(new_nodes,(-1, n_sample**(i+1) )))
		return result


	def get_neighbour_featrue_by_id(self, inputs):
		kg,entity = inputs[0], inputs[1]
		tmp = tf.gather(self.gMatrix,K.cast(entity,dtype='int32'))
		result = tf.gather(tmp,K.cast(kg,dtype='int32'),axis=2)
		return result

	def get_featrue_by_depth(self, inputs):
		fMatrix, level = inputs[0],inputs[1]
		result = tf.gather(fMatrix,K.cast(level,dtype='int32'),axis=-2)
		return result

	def get_embedding_by_id(self, entity):
		tmp = K.gather(self.eMatrix,K.cast(entity,dtype='int32'))
		#result = K.reshape(self.eMatrix,(-1,))
		return tmp

	def get_neighbor_info(self,node,neighbor): #optionl: x * neighbor
		neighbor_score = node * neighbor
		neighbor_score = K.reshape(neighbor_score,
			(K.shape(neighbor_score)[0],-1,self.config.sample_size,self.fMatrix.shape[1]))
		result = K.sum(neighbor_score, axis=2)
		return result



class CosineAnnealingScheduler(keras.callbacks.Callback):
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def cosine_decay_with_warmup(global_step,learning_rate_base,total_steps,warmup_learning_rate=0.0,warmup_steps=0,hold_base_rate_steps=0):
	if total_steps < warmup_steps:
		raise ValueError('total_steps must be larger or equal to warmup_steps.')
	#这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
	learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *(global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
	#如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
	if hold_base_rate_steps > 0:
		learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
								 learning_rate, learning_rate_base)
	if warmup_steps > 0:
		if learning_rate_base < warmup_learning_rate:
			raise ValueError('learning_rate_base must be larger or equal to '
							 'warmup_learning_rate.')
		#线性增长的实现
		slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
		warmup_rate = slope * global_step + warmup_learning_rate
		#只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
		learning_rate = np.where(global_step < warmup_steps, warmup_rate,
								 learning_rate)
	return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
	def __init__(self,
				 learning_rate_base,
				 total_steps,
				 global_step_init=0,
				 warmup_learning_rate=0.0,
				 warmup_steps=0,
				 hold_base_rate_steps=0,
				 verbose=0):
		super(WarmUpCosineDecayScheduler, self).__init__()
		self.learning_rate_base = learning_rate_base
		self.total_steps = total_steps
		self.global_step = global_step_init
		self.warmup_learning_rate = warmup_learning_rate
		self.warmup_steps = warmup_steps
		self.hold_base_rate_steps = hold_base_rate_steps
		self.verbose = verbose
		#learning_rates用于记录每次更新后的学习率，方便图形化观察
		self.learning_rates = []
	#更新global_step，并记录当前学习率
	def on_batch_end(self, batch, logs=None):
		self.global_step = self.global_step + 1
		lr = K.get_value(self.model.optimizer.lr)
		self.learning_rates.append(lr)
	#更新学习率
	def on_batch_begin(self, batch, logs=None):
		lr = cosine_decay_with_warmup(global_step=self.global_step,
									  learning_rate_base=self.learning_rate_base,
									  total_steps=self.total_steps,
									  warmup_learning_rate=self.warmup_learning_rate,
									  warmup_steps=self.warmup_steps,
									  hold_base_rate_steps=self.hold_base_rate_steps)
		K.set_value(self.model.optimizer.lr, lr)
		if self.verbose > 0:
			print('\nBatch %05d: setting learning '
				  'rate to %s.' % (self.global_step + 1, lr))



class ConcatAggregator(Layer):
	def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
				 **kwargs):
		super(ConcatAggregator, self).__init__(**kwargs)
		if activation == 'relu':
			self.activation = K.relu
		elif activation == 'tanh':
			self.activation = K.tanh
		else:
			raise ValueError(f'`activation` not understood: {activation}')
		self.initializer = initializer
		self.regularizer = regularizer

	def build(self, input_shape):
		ent_embed_dim = input_shape[0][-1]
		neighbor_embed_dim = input_shape[1][-1]
		#print(f'aggtrgator shape {ent_embed_dim} {neighbor_embed_dim}')
		self.w = self.add_weight(name=self.name + '_w',
								 shape=(ent_embed_dim+neighbor_embed_dim, ent_embed_dim),
								 initializer=self.initializer, regularizer=self.regularizer)
		self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
								 initializer='zeros')
		super(ConcatAggregator, self).build(input_shape)

	def call(self, inputs, **kwargs):
		entity, neighbor = inputs
		return self.activation(K.dot(K.concatenate([entity, neighbor]), self.w) + self.b)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'activation': self.activation,
			'initializer': self.initializer,
			'regularizer':self.regularizer,
			})
		return config


class ProductAggreator(Layer):
		def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
				 **kwargs):
			super(ProductAggreator,self).__init__()
			if activation == 'relu':
				self.activation = K.relu
			elif activation == 'tanh':
				self.activation = K.tanh
			else:
				raise ValueError(f'`activation` not understood: {activation}')
			self.initializer = initializer
			self.regularizer = regularizer

		def build(self,input_shape):
			dim1 = input_shape[0][-1]
			dim2 = input_shape[1][-1]
			#print(f'test, aggreator dim1:{dim1}, dim2:{dim2}')
			self.w = self.add_weight(name=self.name + '_w',shape=(dim1,dim2),
								 initializer=self.initializer, regularizer=self.regularizer)
			self.b = self.add_weight(name=self.name + '_b',shape=(dim1,),
								 initializer=self.initializer, regularizer=self.regularizer)
			super(ProductAggreator, self).build(input_shape)

		def call(self, inputs, **kwargs):
			entity, neighbor = inputs
			return self.activation(K.dot(neighbor, self.w) + self.b)

		def compute_output_shape(self, input_shape):
			return input_shape[0]

class SumAggregator(Layer):
	def __init__(self, activation: str ='relu', initializer='glorot_normal', regularizer=None,
				 **kwargs):
		super(SumAggregator, self).__init__(**kwargs)
		if activation == 'relu':
			self.activation = K.relu
		elif activation == 'tanh':
			self.activation = K.tanh
		else:
			raise ValueError(f'`activation` not understood: {activation}')
		self.initializer = initializer
		self.regularizer = regularizer

	def build(self, input_shape):
		ent_embed_dim = input_shape[0][-1]
		self.w = self.add_weight(name=self.name+'_w', shape=(ent_embed_dim, ent_embed_dim),
								 initializer=self.initializer, regularizer=self.regularizer)
		self.b = self.add_weight(name=self.name+'_b', shape=(ent_embed_dim,), initializer='zeros')
		super(SumAggregator, self).build(input_shape)

	def call(self, inputs, **kwargs):
		entity, neighbor = inputs
		return self.activation(K.dot((entity + neighbor), self.w) + self.b)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

