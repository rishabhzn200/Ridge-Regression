import numpy as np
import matplotlib.pyplot as plt



class RidgeRegression:

    def __init__(self):
        self.data = None
        self.x = None
        self.y = None

        pass

    def LoadDataSet(self,filename):
        data = np.load(filename)
        dim1, dim2 = np.shape(data)
        self.data = data
        return self.data
        # self.x = data[:, 0]
        # self.y = data[:, 1]
        # return (self.x, self.y)

    def GetX(self, data):
        return data[:, 0]

    def GetY(self, data):
        return data[:, 1]

    def CreateHigherDimensionData(self, x, degree=None):
        degree = [15] if degree is None else degree

        x = x.reshape(x.shape[0], 1)

        multiplier = x #always multiply the result with multiplier

        xlist = np.ones(x.shape[0]).reshape(x.shape[0], 1)
        xlist = np.append(xlist, np.array(x), axis=1)

        for d in range(degree-1):
            x = np.multiply(x, multiplier)
            xlist = np.append(xlist, x, axis=1)

        #abc = xlist.T
        return xlist

        #c = 20



    def EvaluateTheta(self, xtrain, ytrain, learningRate):
        theta = np.matmul(np.linalg.inv(np.matmul(xtrain.T, xtrain) + np.multiply(learningRate ** 2 , np.identity(xtrain.shape[1])) ) , np.matmul(xtrain.T, ytrain)   )

        p1 = np.matmul(xtrain.T, xtrain)
        p2 = np.multiply(learningRate ** 2 , np.identity(xtrain.shape[1]))
        p3 = np.matmul(xtrain.T, ytrain)

        p4 = np.linalg.inv(p1 + p2)
        p5 = np.matmul(p4, p3)

        return theta
        pass


    def EvaluateLoss(self, x, y, theta, learningRate):

        p1 = y - np.matmul(x, theta)
        p2 = np.multiply(learningRate**2, np.matmul(theta.T, theta))

        loss = np.matmul(p1.T, p1) + p2
        return loss


    def SplitIntoTrainValidationData(self, x, y, numFold, foldIteration):

        batchSize = int(x.shape[0] / numFold)

        xtrain = None #np.empty((1,x.shape[1]))
        ytrain = None #np.empty((1,y.shape[1]))
        xtest = None #np.empty((1,x.shape[1]))
        ytest = None #np.empty((1,y.shape[1]))


        for currFoldIter in range(numFold):

            startIndex = currFoldIter * batchSize
            endIndex = (currFoldIter + 1) * batchSize

            if foldIteration == currFoldIter:
                #populate xtest and ytest
                try:
                    xtest = np.append(xtest, x[startIndex:endIndex,:], axis=0)
                    ytest = np.append(ytest, y[startIndex:endIndex,:], axis=0)
                except:
                    xtest = x[startIndex:endIndex, :]
                    ytest = y[startIndex:endIndex, :]
                pass
            else:
                #populate xtrain and ytrain
                try:
                    xtrain = np.append(xtrain, x[startIndex:endIndex, :], axis=0)
                    ytrain = np.append(ytrain, y[startIndex:endIndex, :], axis=0)
                except:
                    xtrain = x[startIndex:endIndex, :]
                    ytrain = y[startIndex:endIndex, :]
                pass

        return (xtrain, ytrain, xtest, ytest)

    def CrossValidation(self, x, y, learningRate, fold=5):

        a = 20

        #combine data using np.append(x,y,axis=1) and permute it
        #np.random.seed(2)
        data_perm = np.random.permutation(np.append(x,y,axis=1))

        dpt = data_perm.T

        x = data_perm[:,:x.shape[1]] #16 cols
        xT = x.T
        y = data_perm[:,-1] #last col which is 17th col in data perm
        yT = y.T

        y = y.reshape(y.shape[0], 1)


        trainErrorList = np.array([])
        validationErrorList = np.array([])

        #Split the data into training and validation sets. 5 folds on x and y
        for currFold in range(fold):
            #currFold can be 0,1,2,3,4
            xtrain, ytrain, xval, yval = self.SplitIntoTrainValidationData(x,y, fold, currFold)

            #Find theta
            theta = self.EvaluateTheta(xtrain, ytrain, learningRate)


            trainError = self.EvaluateLoss(xtrain, ytrain, theta, learningRate)
            validationError = self.EvaluateLoss(xval, yval, theta, learningRate)

            try:
                trainErrorList = np.append(trainErrorList, trainError)
            except:
                trainErrorList = trainError

            try:
                validationErrorList = np.append(validationErrorList, validationError)
            except:
                validationErrorList = validationError

        averageTrainError = np.average(trainErrorList)
        averageValidationError = np.average(validationErrorList)

        c = 10

        return (averageTrainError, averageValidationError)

        pass

    def RidgeRegression(self, data, fold, lambd):
        #TODO x need to be extrapolated
        #TODO for all values of lambda
        #TODO for 5  folds. Find error and average them
        #TODO Plot the average cross validation error and train error w.r.t λ.
        #TODO Select the best λ and estimate θ and plot the polynomial curve to fit the train data.

        TrainErrList = []
        ValidationErrList = []

        for learningRate in lambd:



            x = self.GetX(data)
            y = self.GetY(data)
            y = y.reshape(y.shape[0], 1)

            x15plus1 = self.CreateHigherDimensionData(x, degree=15)


            #Perform Cross Validation and get the average training and validation error
            train_error, val_error = self.CrossValidation(x15plus1, y, learningRate, fold=5)

            TrainErrList.append(train_error)
            ValidationErrList.append(val_error)

        minValidationError = min(ValidationErrList)
        from operator import itemgetter
        minIndex = min(enumerate(ValidationErrList), key=itemgetter(1))[0]

        bestLambd = lambd[minIndex]

        self.PlotFigure(lambd, TrainErrList, ValidationErrList, bestLambd)



        #Find theta for the entire data using the best lambda
        xnew = self.GetX(data)
        ynew = self.GetY(data)
        ynew = ynew.reshape(ynew.shape[0], 1)
        xnewphi = self.CreateHigherDimensionData(xnew, degree=15)
        thetaBest = self.EvaluateTheta(xnewphi, ynew, bestLambd)
        loss = self.EvaluateLoss(xnewphi, ynew, thetaBest, learningRate)
        yhat = np.matmul(xnewphi, thetaBest)

        points = plt.scatter(xnew, ynew, c='red', s=10)
        regLine, = plt.plot(xnew, yhat, color='blue')
        plt.title("RIDGE Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("4_PolynomialFit")
        plt.show()
        plt.gcf().clear()


        d = 20

        pass


    def PlotFigure(self, lambd, train, val, bestLambd):
        a = train[0]

        train_err, = plt.plot(lambd, train, color='red')
        plt.scatter(lambd, train, color='red', s=10)
        plt.title("Error VS learning rate")
        test_error, = plt.plot(lambd, val, color='green')
        plt.scatter(lambd, val, color='green', s=10)
        plt.xlabel("Lambda")
        plt.ylabel("Error")
        plt.legend((train_err, test_error), ("Train Error", "Cross Validation Error"))
        plt.text(7,7,"Optimal Lambda = " + str(bestLambd))
        plt.savefig("3_ErrorVSLambda")
        plt.show()
        plt.gcf().clear()
        pass

if __name__ == "__main__":

    ridgeReg = RidgeRegression()

    data = ridgeReg.LoadDataSet("linRegData.npy")

    x = ridgeReg.GetX(data)
    y = ridgeReg.GetY(data)

    lambd = [0.01, 0.05, 0.1, 0.5, 1.0, 5, 10]

    ridgeReg.RidgeRegression(data, fold=5, lambd=lambd)

    ones = np.ones(100).reshape(100,1)
    x = x.reshape(x.shape[0], 1)
    x_new = np.append(ones, x, axis=1)

    # (x'x)-1 * x'y

    cov = np.matmul(x_new.T , x_new)
    covIverse = np.linalg.inv(cov)


    thetaHat = np.matmul(covIverse, np.matmul(x_new.T, y))


    yHatList = np.matmul(x_new, thetaHat)
