import os
import sys 
import re
from tqdm import tqdm

def check1(fp,p1_index=0,p2_index=1):

	print(f'check duplicat pair of {fp}\n')
	temp_data = ""
	ppi_dict = {}
	lineCount = 0
	dupCount = 0
	skip_head = True
	for line in tqdm(open(fp)):
		lineCount += 1

		if skip_head:
				skip_head = False
				continue		
		line = line.strip().split('\t')
		if line[p1_index] < line[p2_index]:
			temp_data = line[p1_index] + "__" + line[p2_index]
		else:
			temp_data = line[p2_index] + "__" + line[p1_index]

		print(temp_data)
		if temp_data not in ppi_dict.keys():
			ppi_dict[temp_data] = 0
		else:
			dupCount += 1
	
	return

def GetSeqId(fName=""):
#read seq with the following foramt in each line: id seq 
	if not os.path.isfile(fName):
		print("source file does not exisit, name {}\n".format(fName))
		exit()
	sFile = open(fName,'r')
	idDict = dict() #index is id, value is numeric number
	sFile.readline()
	class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}
	count = 0
	while True:
		line = sFile.readline()
		if not line:
			break
		tmp = re.split(',|\t',line)
		if tmp[0] not in idDict:
			idDict[tmp[0]] = count
			count +=1
		if tmp[1] not in idDict:
			idDict[tmp[1]] = count
			count += 1
	sFile.close()
	return idDict

def WriteSeqById(idDict,fName = "",oName = ""):

	sFile = open(fName,'r')
	oFile = open(oName,'w')
	count = 0
	while True:
		newLine = sFile.readline()
		if not newLine:
			break
		line = newLine.strip('\n')
		tmp = re.split(',|\t',line)
		if tmp[0] in idDict: #seq ID
			oFile.write(newLine)

	oFile.close()
	sFile.close()
	return 

if __name__ == "__main__":
	mode = sys.argv[1]
	if mode == '-g':
		print('generate sequence seq list based on interaction list')
		InterFname = sys.argv[2]
		SeqFname = sys.argv[3]
		OutFname = sys.argv[4]
		SeqId = GetSeqId(InterFname)
		WriteSeqById(SeqId,SeqFname,OutFname)



	#fp = sys.argv[2]
	#check1(fp) #check duplicate key
