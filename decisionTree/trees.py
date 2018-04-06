# coding=utf-8

from math import log
import operator


def calShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVect in dataSet:
		currentLabel = featVect[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


def createDataSet():
	dataSet = [ [1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']   ]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels



"""
>>> myDat, labels = createDataSet()
>>> myDat
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
>>> splitDataSet(myDat, 0, 1)
[[1, 'yes'], [1, 'yes'], [0, 'no']]
>>> splitDataSet(myDat, 0, 0)
[[1, 'no'], [1, 'no']]
"""
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVect in dataSet:
		if featVect[axis] == value:
			reducedFeatVect = featVect[:axis]
			reducedFeatVect.extend(featVect[axis+1:])
			retDataSet.append(reducedFeatVect)
	return retDataSet






"""
>>> myDat, labels = createDataSet()
>>> chooseBestFeatureToSplit(myDat)
0
>>> myDat
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
"""
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calShannonEnt(dataSet)
	bestInfoGain , bestFeature = 0.0,  -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature


def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]



def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet\
						(dataSet, bestFeat, value), subLabels)
	return myTree

