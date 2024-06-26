#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
import numpy as np
def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                if sequence[i] not in AADict or sequence[i + g + 1] not in AADict or sequence[i + 2 * g + 2] not in AADict:
                    print(f'{sequence[i]} {sequence[i+1]} {sequence[i + g + 1]} {sequence[i + 2 * g + 2]} ')
                fea = AADict[sequence[i]]+'.'+AADict[sequence[i + g + 1]]+'.'+AADict[sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res

# extract features from the input sequence list
#read per file
def CalCJ(seqList, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'CX'  #SHS148k contains a subsequence 'XD' 
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = ['#', 'label']
    for f in features:
        header.append(f)
    encodings.append(header)

    result = []
    for seq in seqList:
        result.append(CalculateKSCTriad(seq, 0, features, AADict))
    return np.array(result,dtype=float)


#read per file
def CTriad(fastas, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = ['#', 'label']
    for f in features:
        header.append(f)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        if len(sequence) < 3:
            print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
            return 0
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return encodings

