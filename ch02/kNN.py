import numpy as np
import operator

"""Constructor
>>> group,labels = kNN.createDataSet()
>>> group
array([[ 1. , 1.1],
[ 1. , 1. ],
[ 0. , 0. ],
[ 0. , 0.1]])
>>> labels
['A', 'A', 'B', 'B']
"""
def createDataSet():
	group = np.array([ [1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1] ])

	labels = ['A', 'A', 'B', 'B']
	return group, labels

"""Implement K-nearest neighbors"""
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]		# group.shape is (4, 2)
	# Calculate the distance:
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	# to find k-nearest points:
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	#Sort
	sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
	return sortedClassCount[0][0]

"""
>>> datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
>>> datingDataMat	# the result is not the only one
array([[ 7.29170000e+04, 7.10627300e+00, 2.23600000e-01],
[ 1.42830000e+04, 2.44186700e+00, 1.90838000e-01],
[ 7.34750000e+04, 8.31018900e+00, 8.52795000e-01],
...,
[ 1.24290000e+04, 4.43233100e+00, 9.24649000e-01],
[ 2.52880000e+04, 1.31899030e+01, 1.05013800e+00],
[ 4.91800000e+03, 3.01112400e+00, 1.90663000e-01]])

>>> datingLabels[0:20]
[3, 2, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3]
"""
def file2matrix(filename):
	fr = open(filename)
	arrayOfLines = fr.readlines()
	numberOflines = len(arrayOfLines)
	returnMat = np.zeros((numberOflines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector


"""Data-normalizing :
newValue = (oldValue - minVals) / (maxVals - minVals)"""
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	return normDataSet, ranges, minVals


def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],  \
						datingLabels[numTestVecs:m], 3)

		print("the classifier came back with: %d, the real answer is: %d"  \
							% (classifierResult, datingLabels[i]))

		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print("the total error rate is: %f" %(errorCount / float(numTestVecs)))


	
"""
>>> kNN.classifyPerson ()
Percentage of time spent playing video games?10
frequent flier miles earned per year?10000
liters of ice cream consumed per year?0.5
You will probably like this person:  in small does
"""
def classifyPerson():
	resultList = ['not at all', 'in small does', 'in large doses']
	percentTats = float(input("Percentage of time spent playing video games?"))
	ffMiles = float(input('frequent flier miles earned per year?'))
	iceCream = float(input('liters of ice cream consumed per year?'))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = np.array([ffMiles, percentTats, iceCream])
	classifierResult = classify0( (inArr - minVals) / ranges, normMat, datingLabels, 3)
	print('You will probably like this person: ', resultList[classifierResult - 1] )

