import numpy as np
import math
import matplotlib.pyplot as plt
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

best_costs_test_001 = []
best_costs_test_01 = []
best_costs_test_1 = []
cost_test_001_min = -1
cost_test_01_min = -1
cost_test_1_min = -1
best_theta_001 = []
best_theta_01 = []
best_theta_1 = []
cost_cost_history_001 =[]
cost_cost_history_01 = []
cost_cost_history_1 = []
rmse_train_001 = []
rmse_test_001 = []
rmse_train_01 = []
rmse_test_01 = []
rmse_train_1 = []
rmse_test_1 = []

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

    def cost_funct(theta, X, Y):
        m = test_count
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum(np.square(predictions-Y))
        return cost

    def grad_descent(X, Y, theta, alpha, iterations):
        m = X.shape[0]
        for it in range(iterations):
            prediction = np.dot(X, theta)
            theta = theta - (np.dot(X.transpose(), prediction-Y) * (1/m) * alpha)
            cost_history.append(cost_funct(theta, X_test, Y_test))
        return theta

    initial = np.array([1, 1, 1, 1])
    theta = np.asmatrix(initial)
    theta = theta.transpose()
    cost_history=[]
    theta = grad_descent(X_train, Y_train, theta, 0.001, 2000)
    best_theta_001.append(theta)
    best_costs_test_001.append(cost_funct(theta, X_test, Y_test))
    cost_test_001_min = min(best_costs_test_001)
    cost_cost_history_001.append(cost_history)
    prediction_test = np.dot(X_test, theta)
    prediction_train = np.dot(X_train, theta)
    rmse_test_001.append(math.sqrt(np.mean(np.square((prediction_test-Y_test)))))
    rmse_train_001.append(math.sqrt(np.mean(np.square((prediction_train-Y_train)))))


    theta = np.asmatrix(initial)
    theta = theta.transpose()
    cost_history = []
    theta = grad_descent(X_train, Y_train, theta, 0.01, 1000)
    best_theta_01.append(theta)
    best_costs_test_01.append(cost_funct(theta, X_test, Y_test))
    cost_test_01_min = min(best_costs_test_01)
    cost_cost_history_01.append(cost_history)
    prediction_test = np.dot(X_test, theta)
    prediction_train = np.dot(X_train, theta)
    rmse_test_01.append(math.sqrt(np.mean(np.square((prediction_test-Y_test)))))
    rmse_train_01.append(math.sqrt(np.mean(np.square((prediction_train-Y_train)))))

    theta = np.asmatrix(initial)
    theta = theta.transpose()
    cost_history = []
    theta = grad_descent(X_train, Y_train, theta, 0.1, 1000)
    best_theta_1.append(theta)
    best_costs_test_1.append(cost_funct(theta, X_test, Y_test))
    cost_test_1_min = min(best_costs_test_1)
    cost_cost_history_1.append(cost_history)
    prediction_test = np.dot(X_test, theta)
    prediction_train = np.dot(X_train, theta)
    rmse_test_1.append(math.sqrt(np.mean(np.square((prediction_test-Y_test)))))
    rmse_train_1.append(math.sqrt(np.mean(np.square((prediction_train-Y_train)))))

variance_test_001 = 0
variance_test_01 = 0
variance_test_1 = 0
variance_train_001 = 0
variance_train_01 = 0
variance_train_1 = 0
for i in rmse_test_001:
    variance_test_001 += (i-(sum(rmse_test_001)/20))**2
variance_test_001 /= 20

for i in rmse_test_01:
    variance_test_01 += (i-(sum(rmse_test_01)/20))**2
variance_test_01 /= 20

for i in rmse_test_1:
    variance_test_1 += (i-(sum(rmse_test_1)/20))**2
variance_test_1 /= 20

for i in rmse_train_001:
    variance_train_001 += (i-(sum(rmse_train_001)/20))**2
variance_train_001 /= 20

for i in rmse_train_01:
    variance_train_01 += (i-(sum(rmse_train_01)/20))**2
variance_train_01 /= 20

for i in rmse_train_1:
    variance_train_1 += (i-(sum(rmse_train_1)/20))**2
variance_train_1 /= 20

index = -1
i1=-1
i2=-1
i3=-1
for i in range(len(best_costs_test_001)):
    if best_costs_test_001[i] == cost_test_001_min:
        index = i
        i1=index
        break

plt.plot(cost_cost_history_001[index])
plt.show()

index = -1
for i in range(len(best_costs_test_01)):
    if best_costs_test_01[i] == cost_test_01_min:
        index = i
        i2=index
        break
plt.plot(cost_cost_history_01[index])
plt.show()

index = -1
for i in range(len(best_costs_test_1)):
    if best_costs_test_1[i] == cost_test_1_min:
        index = i
        i3=index
        break
plt.plot(cost_cost_history_1[index])
plt.show()

print("Minimum Value of Cost Function obtained with alpha as 0.001 on testing data = " + str(cost_test_001_min))
print("Minimum RMSE obtained from the 20 training data sets with alpha as 0.001 = " + str(min(rmse_train_001)))
print("Minimum RMSE obtained from the 20 testing data sets with alpha as 0.001 = " + str(min(rmse_test_001)))
print("Mean of RMSE of training data = " + str(sum(rmse_train_001)/20))
print("Variance of RMSE of training data = " + str(variance_train_001))
print("Mean of RMSE of testing data = " + str(sum(rmse_test_001)/20))
print("Variance of RMSE of testing data = " + str(variance_test_001))
print("Best value of theta matrix obtained: ")
print(best_theta_001[i1])
print("")
print("")
print("")
print("Minimum Value of Cost Function obtained with alpha as 0.01 on testing data = " + str(cost_test_01_min))
print("Minimum RMSE obtained from the 20 training data sets with alpha as 0.01 = " + str(min(rmse_train_01)))
print("Minimum RMSE obtained from the 20 testing data sets with alpha as 0.01 = " + str(min(rmse_test_01)))
print("Mean of RMSE of training data = " + str(sum(rmse_train_01)/20))
print("Variance of RMSE of training data = " + str(variance_train_01))
print("Mean of RMSE of testing data = " + str(sum(rmse_test_01)/20))
print("Variance of RMSE of testing data = " + str(variance_test_01))
print("Best value of theta matrix obtained: ")
print(best_theta_01[i1])
print("")
print("")
print("Minimum Value of Cost Function obtained with alpha as 0.1 on testing data = " + str(cost_test_1_min))
print("Minimum RMSE obtained from the 20 training data sets with alpha as 0.1 = " + str(min(rmse_train_1)))
print("Minimum RMSE obtained from the 20 testing data sets with alpha as 0.1 = " + str(min(rmse_test_1)))
print("Mean of RMSE of training data = " + str(sum(rmse_train_1)/20))
print("Variance of RMSE of training data = " + str(variance_train_1))
print("Mean of RMSE of testing data = " + str(sum(rmse_test_1)/20))
print("Variance of RMSE of testing data = " + str(variance_test_1))
print("Best value of theta matrix obtained: ")
print(best_theta_1[i1])