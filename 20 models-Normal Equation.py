import numpy as np
import math
age=[]
bmi=[]
children=[]
charges=[]
one=[]
count=0
f = open('insurance.txt', 'r')
for i in f:
    if count>0:
        words = i.split(',')
        one.append(1)
        age.append(int(words[0]))
        bmi.append(float(words[1]))
        children.append(int(words[2]))
        charges.append(float(words[3]))
    count+=1



age_max=max(age)
age_min=min(age)
bmi_max=max(bmi)
bmi_min=min(bmi)
children_max=max(children)
children_min=min(children)


for x in range(len(age)):
    age[x] = (age[x]-age_min)/(age_max-age_min)
for x in range(len(bmi)):
    bmi[x]= (bmi[x]-bmi_min)/(bmi_max-bmi_min)
for x in range(len(children)):
    children[x]= (children[x]-children_min)/(children_max-children_min)

rmse_test_history = []
rmse_train_history = []
theta_history = []

for qwerty in range(20):
    q = np.array([one, age, bmi, children, charges])
    q = np.transpose(q)
    np.random.shuffle(q)
    w = q
    w = np.delete(w, 4, axis=1)
    Data = np.asmatrix(q)
    Data_transpose = Data.transpose()
    Y = np.delete(Data, [0,1,2,3], axis=1)
    X = np.asmatrix(w)
    train_count = math.floor(count*0.7)
    test_count = count - train_count
    Y_train, Y_test = np.split(Y, [train_count])
    X_train, X_test = np.split(X, [train_count])
    X_train_transpose = X_train.transpose()
    Mult = np.matmul(X_train_transpose, X_train)
    Multinv=np.linalg.inv(np.matrix(Mult))
    Multinv = np.asmatrix(Multinv)
    Product = np.matmul(Multinv, X_train_transpose)
    Result = np.matmul(Product, Y_train)
    theta_history.append(Result)
    Y_bar_test = np.matmul(X_test, Result)
    Y_bar_train = np.matmul(X_train, Result)
    Error_test = Y_test-Y_bar_test
    Error_train = Y_train-Y_bar_train
    rmse_train=0
    rmse_test=0
    for i in Error_test:
        rmse_test+=i*i
    for i in Error_train:
        rmse_train+=i*i
    rmse_train = rmse_train/train_count
    rmse_train = math.sqrt(rmse_train)
    rmse_test = rmse_test/test_count
    rmse_test = math.sqrt(rmse_test)

    rmse_test_history.append(rmse_test)
    rmse_train_history.append(rmse_train)

#Printing values related to errors obtained in training datasets
rmse_train_min = min(rmse_train_history)
print("Minimum RMSE obtained from the 20 training data sets = " + str(rmse_train_min))
rmse_train_mean = sum(rmse_train_history)/20
print("Mean of RMSE of training data sets = " + str(rmse_train_mean))
variance_train = 0
for i in rmse_train_history:
    variance_train+=(i-rmse_train_mean)**2
variance_train/=20
print("Variance of RMSE of training data sets = " + str(variance_train))

#Printing values related to errors obtained in testing datasets
rmse_test_min = min(rmse_test_history)
print("Minimum RMSE obtained from the 20 testing data sets = " + str(rmse_test_min))
rmse_test_mean = sum(rmse_test_history)/20
print("Mean of RMSE of testing data sets = " + str(rmse_test_mean))
variance_test = 0
for i in rmse_test_history:
    variance_test+=(i-rmse_test_mean)**2
variance_test/=20
print("Variance of RMSE of testing data sets = " + str(variance_test))

#Printing the value of best theta obtained
index = -1
for i in range(len(rmse_train_history)):
    if rmse_train_history[i] == rmse_train_min:
        index = i
        break
print("Best value for the theta matrix obtained is: ")
print(theta_history[index])