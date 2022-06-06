import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time


totals = []
trainAverage = []
trainClasses = []
trainingData = []
testClasses = []
testingData = []

def getFeatures(file, type):
    data = []
    
    for line in file:
        data.append(line.rstrip("\n").split(","))

    del data[0]


    i = 0
    while i < len(data):
        features = []
        if type=="train":
            trainClasses.append(int(data[i][0]))
        elif type=="test":
            testClasses.append(int(data[i][0]))
        total = 0 #stores the sum of pixels for each sample
        zeros = 0 #Stores the number of zeros
        count = 1 #Iterator
        avg = 0 #Average pixel value
        nonzero = 0 #Number of nonzero pixels
        largestConsecutiveNonZero = 0 #Largest number of consecutive nonzero pixels for each sample
        firstAdjacentNonzeros = 0 #First pixel >250 with adjacent pixels greater than 0
        lastAdjacentNonzeros = 0
        count250 = 0 #Number of pixels with value greater than 250
        countConsecutive = 0 
        dataLen = len(data[i])
        countRow = 1 #Tracks current iteration of the 28x28 image
        numNonZero = 0 #Stores the total number of consecutive pixels greater than 0
        foundFirst = False
        first250x= 0 #Stores the first occurance of value greater than 250
        last250x = 0 #Stores the last occurance of value greater than 250
        
        while count<dataLen:
            curr = int(data[i][count]) #Current pixel value
            if curr==0:
                zeros+=1
                largestConsecutiveNonZero=nonzero
            if countRow==28:
                countRow=1
                numNonZero+=countConsecutive
            elif curr>250:
                count250+=1
                if data[i][count+1] != 0:
                    countConsecutive+=1
                if foundFirst==False:
                    first250x=count
                    foundFirst=True
                    if i<dataLen and firstAdjacentNonzeros==0 and int(data[i][count-1])>0 and int(data[i][count+1])>0 and int(data[i+1][count])>0 and int(data[i][count])>0 \
                        and int(data[i+1][count+1])>0 and int(data[i][count+1])>0  and int(data[i+1][count-1])>0 and int(data[i][count-1])>0:

                        firstAdjacentNonzeros=count
                last250x=count
                if i<dataLen and int(data[i][count-1])>0 and int(data[i][count+1])>0 and int(data[i+1][count])>0 and int(data[i][count])>0 \
                        and int(data[i+1][count+1])>0 and int(data[i][count+1])>0  and int(data[i+1][count-1])>0 and int(data[i][count-1])>0:

                        lastAdjacentNonzeros=count
                
            else:
                nonzero+=1
            countRow+=1
            total+=curr
            count+=1
        avg+=total/dataLen
        features.append(zeros)
        features.append(total)
        features.append(avg)
        features.append(largestConsecutiveNonZero)
        features.append(firstAdjacentNonzeros)
        features.append(count250)
        features.append(countConsecutive)
        features.append(numNonZero)
        #Distnce between first 250+ pixel and last 250+ pixel
        firstDist = math.sqrt((int(data[i][last250x])-int(data[i][first250x]))**2+(last250x-first250x)**2) 
        features.append(firstDist)
        #Distance between first 250+ pixel and first 250 pixel with adjacent pixels >250
        firstAdjFirstDist = math.sqrt((int(data[i][first250x])-int(data[i][firstAdjacentNonzeros]))**2+(first250x-firstAdjacentNonzeros)**2)
        features.append(firstAdjFirstDist)
        FirstAdjLastDist = math.sqrt((int(data[i][last250x])-int(data[i][firstAdjacentNonzeros]))**2+(last250x-firstAdjacentNonzeros)**2)
        features.append(FirstAdjLastDist)
        #Distance between first 250 pixel and last pixel with adjacent pixels >250
        LastAdjFirstDist = math.sqrt((int(data[i][first250x])-int(data[i][lastAdjacentNonzeros]))**2+(first250x-lastAdjacentNonzeros)**2)
        features.append(LastAdjFirstDist)

        LastAdjLastDist = math.sqrt((int(data[i][last250x])-int(data[i][lastAdjacentNonzeros]))**2+(last250x-lastAdjacentNonzeros)**2)
        features.append(LastAdjLastDist)
        #Distance between first and last pixels with adjacent pixels >250
        LastAdjFirstAdjDist = math.sqrt((int(data[i][lastAdjacentNonzeros])-int(data[i][firstAdjacentNonzeros]))**2+(lastAdjacentNonzeros-firstAdjacentNonzeros)**2)
        features.append(LastAdjFirstAdjDist)


        #Distance between last 250 pixel and the images midpoint
        lastMidDist = math.sqrt((int(data[i][last250x])-int(data[i][dataLen//2]))**2+(last250x-dataLen/2)**2)
        features.append(lastMidDist)
        firstMidDist = math.sqrt((int(data[i][first250x])-int(data[i][dataLen//2]))**2+(first250x-dataLen/2)**2)
        features.append(firstMidDist)
        #Distance between first adjacent pixel and midpoint
        firstAdjMidDist = math.sqrt((int(data[i][dataLen//2])-int(data[i][firstAdjacentNonzeros]))**2+(dataLen/2-firstAdjacentNonzeros)**2)
        features.append(firstAdjMidDist)
        lastAdjMidDist = math.sqrt((int(data[i][lastAdjacentNonzeros])-int(data[i][dataLen//2]))**2+(lastAdjacentNonzeros-dataLen/2)**2)
        features.append(lastAdjMidDist)
        #Distance between first 250 pixel and the first pixel in the image
        firstElementDist = math.sqrt((int(data[i][first250x])-int(data[i][0]))**2+(first250x-0)**2)
        features.append(firstElementDist)
        lastElementDist = math.sqrt((int(data[i][last250x])-int(data[i][dataLen-1]))**2+(last250x-(dataLen-1))**2)
        features.append(lastElementDist)
        



        if type=="train":
            trainingData.append(features)
        elif type=="test":
            testingData.append(features)
        i+=1
    
    

file = open("data\mnist_train.csv", "r")
start = time.time()
getFeatures(file, "train")

file = open("data\mnist_test.csv", "r")
getFeatures(file, "test")

end = time.time()
print("Time to extract features: "+str(end-start))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(trainingData, trainClasses)
tree = dt.predict(testingData)
tree_acc = accuracy_score(testClasses, tree)
tree_prec = precision_score(testClasses, tree, average="macro")
tree_recall = recall_score(testClasses, tree, average="macro")

nb = GaussianNB()
nb.fit(trainingData, trainClasses)
nbr = nb.predict(testingData)
nb_acc = accuracy_score(testClasses, nbr)
nb_prec = precision_score(testClasses, nbr, average="macro")
nb_recall = recall_score(testClasses, nbr, average="macro")


perceptron = Perceptron(tol=1e-3, random_state=0)
perceptron.fit(trainingData, trainClasses)
perceptron_results = perceptron.predict(testingData)
ann_acc = accuracy_score(testClasses, perceptron_results)
ann_prec = precision_score(testClasses, perceptron_results, average="macro")
ann_recall = recall_score(testClasses, perceptron_results, average="macro")

lgr = LogisticRegression(random_state=0)
lgr.fit(trainingData, trainClasses)
lgr_results = lgr.predict(testingData)
lgr_acc = accuracy_score(testClasses, lgr_results)
lgr_prec = precision_score(testClasses, lgr_results, average="macro")
lgr_recall = recall_score(testClasses, lgr_results, average="macro")

supVec = svm.SVC()
supVec.fit(testingData, testClasses)
pred = supVec.predict(testingData)
svm_acc = accuracy_score(testClasses, pred)
svm_prec = precision_score(testClasses, pred, average="macro")
svm_recall = recall_score(testClasses, pred, average="macro")


print('Decision Tree Scores: ')
print("   Accuracy: "+str(tree_acc * 100)+"%")
print("   Recall: "+ str(tree_recall * 100)+"%")
print("   Precision: "+str(tree_prec * 100)+"%\n")

print('Naive Bayes Scores: ')
print("   Accuracy: "+str(nb_acc * 100)+"%")
print("   Recall: "+ str(nb_recall * 100)+"%")
print("   Precision: "+str(nb_prec * 100)+"%\n")

print('ANN Scores: ')
print("   Accuracy: "+str(ann_acc * 100)+"%")
print("   Recall: "+ str(ann_recall * 100)+"%")
print("   Precision: "+str(ann_prec * 100)+"%\n")

print('Logistic Regression Scores: ')
print("   Accuracy: "+str(lgr_acc * 100)+"%")
print("   Recall: "+ str(lgr_recall * 100)+"%")
print("   Precision: "+str(lgr_prec * 100)+"%\n")

print('SVM Score: ')
print("   Accuracy: "+str(svm_acc * 100)+"%")
print("   Recall: "+ str(svm_recall * 100)+"%")
print("   Precision: "+str(svm_prec * 100)+"%\n")
