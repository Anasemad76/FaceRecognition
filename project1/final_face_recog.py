from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os
from time import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sys
from sklearn import datasets
import pandas as pd



def pca(X, alpha):

    # Compute the mean of the data matrix
    X_mean = np.mean(X, axis=0)

    # Compute the centered data matrix
    X_centered = X - X_mean

    # Compute the covariance matrix
    C = np.cov(X_centered.T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort the eigenvectors in descending order of the eigenvalues
    idx = np.absolute(eigenvalues).argsort()[::-1]
    eigenvectors = eigenvectors[:,idx]

    count = 0
    counter_iterator = 0
    for i in idx:
        count += eigenvalues[i] / sum(eigenvalues)
        counter_iterator += 1

        if count >= alpha:

            break

    n_components = counter_iterator
    U = eigenvectors[:, :n_components]
    print("dimension of cenetered data",X_centered.shape,U.shape)
    X_reduced = np.matmul(X,U)
    return X_reduced, U

def lda_bonus(data_matrix,train_label):
    print(len(data_matrix))
    y = 0
    mean_arr=[]
    centered_data=[]
    mean_vector = np.mean(data_matrix, axis=0)

    for i in range(7,287,7):
        data_matrix_modified=[]
        while(y<i):
            data_matrix_modified.append(data_matrix[y]) # collect every 5 rows together
            y=y+1
        mean_arr.append(np.mean(data_matrix_modified, axis=0))
        centered_data.append(np.subtract(data_matrix_modified,mean_arr[i//7 -1]))


    centered_data=np.reshape(centered_data, (280,10304))
    centered_data=np.array(centered_data)


    mean_vector=np.array(mean_vector)
    mean_arr = np.array(mean_arr)
    class_label = train_label
    class_label = np.array(class_label)
    iet = 0
    S_total = np.zeros([10304, 10304])
    x_a = np.zeros([7, 10304])

    for c in range(1,41):
        print("ietration",iet)
        iet=iet+1
        x_a = centered_data[class_label == c]
        S_total += np.matmul(x_a.T,x_a)
        print("B matrix",S_total)





    data_matrix=np.array(data_matrix)
    n_features = data_matrix.shape[1]
    class_label=train_label
    class_label=np.array(class_label)
    s_b=np.zeros((n_features,n_features))
    iet=0
    for c in range(1,41):
        print("iteration",iet)
        iet=iet+1
        x_c = data_matrix[class_label == c]
        mean_c=np.mean(x_c,axis=0)
        mean_c=np.array(mean_c)
        n_c=x_c.shape[0]
        mean_diff=(mean_c-mean_vector.reshape(n_features,1)) #1*10304.reshape-->10304*1
        s_b += n_c * np.matmul(mean_diff,mean_diff.T)
        print("s_b",s_b)





    # Use 39 eigenvectors
    eigen_values,eigen_vectors = np.linalg.eigh( np.matmul(np.linalg.inv(S_total),s_b)  )
    print("eigen vectors",eigen_vectors.shape)

    idx = np.absolute(eigen_values).argsort()[::-1]
    eigenvectors = eigen_vectors[:,idx]

    u=eigenvectors[:,:39] # Projection matrix
    print("U =",u.shape)


    projection_data = np.matmul(data_matrix,u)
    print("projection data",projection_data)

    return projection_data,u




def lda_org(data_matrix,train_label):

    y = 0
    mean_arr=[]
    centered_data=[]
    mean_vector = np.mean(data_matrix, axis=0)

    for i in range(5,205,5):
        data_matrix_modified=[]
        while(y<i):
            data_matrix_modified.append(data_matrix[y]) # collect every 5 rows together
            y=y+1
        mean_arr.append(np.mean(data_matrix_modified, axis=0))
        centered_data.append(np.subtract(data_matrix_modified,mean_arr[i//5 -1]))


    centered_data=np.reshape(centered_data, (200,10304))
    centered_data=np.array(centered_data)


    mean_vector=np.array(mean_vector)
    mean_arr = np.array(mean_arr)
    class_label = train_label
    class_label = np.array(class_label)
    iet = 0
    S_total = np.zeros([10304, 10304])
    x_a = np.zeros([5, 10304])

    for c in range(1,41):
        print("ietration",iet)
        iet=iet+1
        x_a = centered_data[class_label == c]
        S_total += np.matmul(x_a.T,x_a)
        print("B matrix",S_total)





    data_matrix=np.array(data_matrix)
    n_features = data_matrix.shape[1]
    class_label=train_label
    class_label=np.array(class_label)
    s_b=np.zeros((n_features,n_features))
    iet=0
    for c in range(1,41):
        print("iteration",iet)
        iet=iet+1
        x_c = data_matrix[class_label == c]
        mean_c=np.mean(x_c,axis=0)
        mean_c=np.array(mean_c)
        n_c=x_c.shape[0]
        mean_diff=(mean_c-mean_vector.reshape(n_features,1)) #1*10304.reshape-->10304*1
        s_b += n_c * np.matmul(mean_diff,mean_diff.T)
        print("s_b",s_b)





    # Use 39 eigenvectors
    eigen_values,eigen_vectors = np.linalg.eigh( np.matmul(np.linalg.inv(S_total),s_b)  )
    print("eigen vectors",eigen_vectors.shape)

    idx = np.absolute(eigen_values).argsort()[::-1]
    eigenvectors = eigen_vectors[:,idx]

    u=eigenvectors[:,:39] # Projection matrix
    print("U =",u.shape)


    projection_data = np.matmul(data_matrix,u)
    print("projection data",projection_data)

    return projection_data,u




def lda(data_matrix,train_label):
    print(data_matrix.shape)
    y = 0
    mean_arr=[]
    centered_data=[]
    mean_vector = np.mean(data_matrix, axis=0)

    for i in range(200,405,200): # two classes each of 200
        data_matrix_modified=[]
        while(y<i):
            data_matrix_modified.append(data_matrix[y]) # collect every 200 rows together
            y=y+1
        mean_arr.append(np.mean(data_matrix_modified, axis=0))
        centered_data.append(np.subtract(data_matrix_modified,mean_arr[i//200 -1]))


    centered_data=np.reshape(centered_data, (400,10304))
    centered_data=np.array(centered_data)


    mean_vector=np.array(mean_vector)
    mean_arr = np.array(mean_arr)
    class_label = train_label
    class_label = np.array(class_label)
    iet = 0
    S_total = np.zeros([10304, 10304])
    x_a = np.zeros([200, 10304])

    for c in range(0,2):
        print("iteration",iet)
        iet=iet+1
        x_a = centered_data[class_label == c]
        print(len(x_a))
        S_total += np.matmul(x_a.T,x_a)
        print("S_total",S_total)

    data_matrix = np.array(data_matrix)
    n_features = data_matrix.shape[1]
    class_label = train_label
    class_label = np.array(class_label)
    s_b = np.zeros((n_features, n_features))
    iet = 0
    for c in range(0,1):
        print("iteration", iet)
        iet = iet + 1

        print("\n")
        mean_diff = (mean_arr[0].reshape(n_features, 1) - mean_arr[1].reshape(n_features, 1))
        s_b += np.matmul(mean_diff, mean_diff.T)
        print("s_b", s_b)





    # Use eigenvectors
    eigen_values,eigen_vectors = np.linalg.eigh( np.matmul(np.linalg.inv(S_total),s_b)  )
    print("eigen values",len(eigen_values))
    print("eigen vectors",eigen_vectors.shape)
    idx = np.absolute(eigen_values).argsort()[::-1]
    eigenvectors = eigen_vectors[:,idx]

    u=eigenvectors[:,:1] # Projection matrix should be 1 when dealing with a binary classification problem but still the more u the better
    print("U =",u.shape)

    # projection_data = np.dot(centered_data, u)  # projected data
    projection_data = np.matmul(data_matrix,u)
    projection_data_2 = np.dot(centered_data, u)


    return projection_data,u



def generate_data():
    g=input("Press 1 to generate  faces problem and 2 to generate non faces problem\n")
    if g=="1":
        Data = []
        label = []
        for i in range(1, 41):
            images = os.listdir("./att_faces/s" + str(i))
            for image in images:
                img = cv2.imread('./att_faces/s' + str(i) + "/" + image, 0)
                img_flat = np.array(img).ravel()

                subject = int(i)
                Data.append(img_flat)
                label.append(subject)

        test_data = []
        train_data = []

        test_label = []
        train_label = []


        for x in range(400):

            if x % 2 == 0:
                test_data.append(Data[x])
                test_label.append(label[x])
            else:
                train_data.append(Data[x])
                train_label.append(label[x])
        test_data = np.array(test_data)
        train_data = np.array(train_data)

        test_label = np.array(test_label)
        train_label = np.array(train_label)

        return train_data, train_label, test_data,test_label,1


    elif g=="2":
        non_faces_data = []
        non_faces_label=[]
        images = os.listdir("./non_faces/")
        for image_file in images:
            image_path = os.path.join("./non_faces/", image_file)
            image = cv2.imread(image_path)
            img = cv2.resize(image, (92, 112))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flat = np.array(gray).ravel()
            subject = int(0)

            non_faces_data.append(img_flat)
            non_faces_label.append(subject)





        faces_data=[]
        faces_label=[]
        for i in range(1,41):
            images = os.listdir("./att_faces/s"+str(i))
            for image in images:

                img = cv2.imread('./att_faces/s'+str(i)+"/"+image,0)
                img_flat = np.array(img).ravel()
                subject = int(1)

                faces_data.append(img_flat)
                faces_label.append(subject)



        nonfaces_test_data=[]
        nonfaces_train_data=[]

        nonfaces_test_label=[]
        nonfaces_train_label=[]

        for x in range(400):

            if x % 2 == 0:
                nonfaces_test_data.append(faces_data[x])
                nonfaces_test_label.append(faces_label[x])
            else:
                nonfaces_train_data.append(faces_data[x])
                nonfaces_train_label.append(faces_label[x])

        for x in range(400):

            if x % 2 == 0:
                nonfaces_test_data.append(non_faces_data[x])
                nonfaces_test_label.append(non_faces_label[x])
            else:
                nonfaces_train_data.append(non_faces_data[x])
                nonfaces_train_label.append(non_faces_label[x])



        nonfaces_test_data=np.array(nonfaces_test_data)
        nonfaces_train_data=np.array(nonfaces_train_data)

        nonfaces_test_label=np.array(nonfaces_test_label)
        nonfaces_train_label=np.array(nonfaces_train_label)

        return nonfaces_train_data,nonfaces_train_label,nonfaces_test_data,nonfaces_test_label,2

    sys.exit()





def generate_bonus_data():
    Data = []
    label = []
    for i in range(1, 41):
        images = os.listdir("./att_faces/s" + str(i))
        for image in images:
            img = cv2.imread('./att_faces/s' + str(i) + "/" + image, 0)
            img_flat = np.array(img).ravel()

            subject = int(i)
            Data.append(img_flat)
            label.append(subject)

    print(Data)
    test_data = []
    train_data = []
    test_label = []
    train_label = []
    for x in range(400):
        if x % 10 > 2:
            train_data.append(Data[x])
            train_label.append(label[x])
        else:
            test_data.append(Data[x])
            test_label.append(label[x])



    LDA_X_train_reduced, U_LDA = lda_bonus(train_data, train_label)




    LDA_X_test_reduced = np.matmul(test_data,U_LDA)


    knn_lda = KNeighborsClassifier(n_neighbors=1)
    knn_lda.fit(LDA_X_train_reduced,train_label)
    labels_pred = knn_lda.predict(LDA_X_test_reduced)



    # # Compute the accuracy
    acc_lda = np.mean(np.array(labels_pred) ==test_label)
    print(f"accuracy = {acc_lda}")

    return test_data, test_label, train_data, train_label


def lda_call(nonfaces_train_data,nonfaces_train_label,nonfaces_test_data,nonfaces_test_label,inp):
    if inp==1:
        LDA_X_train_reduced, U_LDA = lda_org(nonfaces_train_data, nonfaces_train_label)



    elif inp==2:

        LDA_X_train_reduced, U_LDA = lda(nonfaces_train_data , nonfaces_train_label)


    LDA_X_test_reduced = np.matmul(nonfaces_test_data,U_LDA)


    knn_lda = KNeighborsClassifier(n_neighbors=1)
    knn_lda.fit(LDA_X_train_reduced,nonfaces_train_label)
    labels_pred = knn_lda.predict(LDA_X_test_reduced)



    # # Compute the accuracy
    acc_lda = np.mean(np.array(labels_pred) ==nonfaces_test_label)
    print(f"accuracy = {acc_lda}")

    if inp==2:

        TN, FP, FN, TP = confusion_matrix(nonfaces_test_label, labels_pred).ravel()

        print('True Positive(TP)  = ', TP)
        print('False Positive(FP) = ', FP)
        print('True Negative(TN)  = ', TN)
        print('False Negative(FN) = ', FN)

        accuracy = (TP + TN) / (TP + FP + TN + FN)

        print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))




def pca_call(nonfaces_train_data,nonfaces_train_label,nonfaces_test_data,nonfaces_test_label,inp):
    # call PCA
    alphas = [0.8, 0.85, 0.9, 0.95]
    for alpha in alphas:
        # Project the training set and test set using the same projection matrix
        pca_X_train_reduced, U_PCA = pca(nonfaces_train_data, alpha)
        pca_X_test_reduced = np.matmul(nonfaces_test_data, U_PCA)

        knn_pca = KNeighborsClassifier(n_neighbors=1)
        knn_pca.fit(pca_X_train_reduced,nonfaces_train_label)
        labels_pred_pca = knn_pca.predict(pca_X_test_reduced)

        # # Compute the accuracy
        acc_pca = np.mean(np.array(labels_pred_pca) ==nonfaces_test_label)
        print("alpha =",alpha)
        print(f"accuracy = {acc_pca}")
        print("\n")

        if inp == 2:

            TN, FP, FN, TP = confusion_matrix(nonfaces_test_label, labels_pred_pca).ravel()

            print('True Positive(TP)  = ', TP)
            print('False Positive(FP) = ', FP)
            print('True Negative(TN)  = ', TN)
            print('False Negative(FN) = ', FN)

            accuracy = (TP + TN) / (TP + FP + TN + FN)

            print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))


def lda_plot(nonfaces_train_data,nonfaces_train_label,nonfaces_test_data,nonfaces_test_label):

    LDA_X_train_reduced, U_LDA = lda(nonfaces_train_data, nonfaces_train_label)
    print("Trained data projected successfully \n")

    z=[]
    accuracy=[]
    for i in range(250,401,30):
        LDA_X_test_reduced = np.matmul(nonfaces_test_data[200:i], U_LDA)
        y_pred = []
        for k in range(i-200): # number of tested data
            dists = []
            for j in range(400):
                dists.append(np.linalg.norm(LDA_X_train_reduced[j, :] - LDA_X_test_reduced[k, :]))

            nn_idx = np.argmin(dists)

            y_pred.append(nonfaces_train_label[nn_idx])

        # Compute the accuracy
        acc = np.mean(np.array(y_pred) ==nonfaces_test_label[200:i])
        print(f"accuracy = {acc}")
        accuracy.append(acc)




        z.append(i-200) # +30

    plt.plot(z,accuracy, label='PCA')
    plt.xlabel('number of non faces')
    plt.ylabel('Accuracy')
    plt.title('K-NN Classifier Performance')
    plt.legend()
    plt.show()

    ################################################################################################
    # accuracy using all test array
    # (
    # LDA_X_train_reduced, U_LDA = lda(nonfaces_train_data, nonfaces_train_label)
    # print("Trained data projected successfully \n")
    #
    # z = []
    # accuracy = []
    # for i in range(250, 401, 30):
    #     LDA_X_test_reduced = np.matmul(nonfaces_test_data[0:i], U_LDA)
    #     y_pred = []
    #     for k in range(i):  # number of tested data
    #         dists = []
    #         for j in range(400):
    #             dists.append(np.linalg.norm(LDA_X_train_reduced[j, :] - LDA_X_test_reduced[k, :]))
    #
    #         nn_idx = np.argmin(dists)
    #
    #         y_pred.append(nonfaces_train_label[nn_idx])
    #
    #     # Compute the accuracy
    #     acc = np.mean(np.array(y_pred) == nonfaces_test_label[:i])
    #     print(f"accuracy = {acc}")
    #     accuracy.append(acc)
    #
    #     ###########################################
    #     # knn_lda = KNeighborsClassifier(n_neighbors=1)
    #     # knn_lda.fit(LDA_X_train_reduced, nonfaces_train_label)
    #     # labels_pred = knn_lda.predict(LDA_X_test_reduced)
    #     #
    #     # # # Compute the accuracy
    #     # acc_lda = np.mean(np.array(labels_pred) == nonfaces_test_label)
    #     # print(f"accuracy = {acc_lda}")
    #
    #     # TN, FP, FN, TP = confusion_matrix(nonfaces_test_label, labels_pred).ravel()
    #
    #     # TN, FP, FN, TP = confusion_matrix(nonfaces_test_label,y_pred).ravel()
    #     #
    #     # print('True Positive(TP)  = ', TP)
    #     # print('False Positive(FP) = ', FP)
    #     # print('True Negative(TN)  = ', TN)
    #     # print('False Negative(FN) = ', FN)
    #     #
    #     # accuracy.append((TP + TN) / (TP + FP + TN + FN))
    #
    #     # print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy[-1]))
    #
    #     z.append(i - 200)  # +30
    #
    # plt.plot(z, accuracy, label='PCA')
    # plt.xlabel('number of non faces')
    # plt.ylabel('Accuracy')
    # plt.title('K-NN Classifier Performance')
    # plt.legend()
    # plt.show()
    # )




def lda_large(data_matrix,train_label):
    # data matrix=250*10304 50 faces and 200 non faces
    print(data_matrix.shape)
    y = 0
    mean_arr=[]
    centered_data=[]
    mean_vector = np.mean(data_matrix, axis=0)
    centered_data_1=[]
    centered_data_2=[]
    j=0


    for i in range(50,255,200):
        data_matrix_modified=[]
        while(y<i):
            data_matrix_modified.append(data_matrix[y])
            y=y+1
        mean_arr.append(np.mean(data_matrix_modified, axis=0))
        if j==0:
            centered_data_1.append(np.subtract(data_matrix_modified,mean_arr[j]))
            j+=1
        else:
            centered_data_2.append(np.subtract(data_matrix_modified, mean_arr[j]))






    centered_data_1=np.array(centered_data_1)
    centered_data_1=centered_data_1.reshape(50,10304)

    centered_data_2=np.array(centered_data_2)
    centered_data_2 = centered_data_2.reshape(200, 10304)


    print(centered_data_1.shape,centered_data_2.shape)

    centered_data=np.concatenate((centered_data_1,centered_data_2))

    print(centered_data)


    mean_vector=np.array(mean_vector)
    mean_arr = np.array(mean_arr)
    class_label = train_label
    class_label = np.array(class_label)
    iet = 0
    S_total = np.zeros([10304, 10304])
    x_a = np.zeros([50, 10304])
    x_b = np.zeros([200, 10304])





    x_a=centered_data_1
    S_total = np.matmul(x_a.T, x_a)
    x_b = centered_data_2
    S_total += np.matmul(x_b.T, x_b)

    data_matrix = np.array(data_matrix)
    n_features = data_matrix.shape[1]
    class_label = train_label
    class_label = np.array(class_label)
    s_b = np.zeros((n_features, n_features))
    iet = 0
    for c in range(0,1):
        print("iteration", iet)
        iet = iet + 1

        print("\n")
        mean_diff = (mean_arr[0].reshape(n_features, 1) - mean_arr[1].reshape(n_features, 1))  # 1*10304.reshape-->10304*1
        s_b += np.matmul(mean_diff, mean_diff.T)
        print("s_b", s_b)





    # Use eigenvectors
    eigen_values,eigen_vectors = np.linalg.eigh( np.matmul(np.linalg.inv(S_total),s_b)  )
    print("eigen values",len(eigen_values))
    print("eigen vectors",eigen_vectors.shape)
    idx = np.absolute(eigen_values).argsort()[::-1]
    eigenvectors = eigen_vectors[:,idx]

    u=eigenvectors[:,:1] # Projection matrix should be 1 when dealing with a binary classification problem but still the more u the better
    print("U =",u.shape)


    projection_data = np.matmul(data_matrix,u)



    return projection_data,u


def large_non_faces(nonfaces_train_data,nonfaces_train_label,nonfaces_test_data,nonfaces_test_label):

    non_faces_train_data_modified=np.zeros([250,10304])
    non_faces_test_data_modified=np.zeros([250,10304])
    non_faces_train_label_modified=np.zeros([250,10304])
    non_faces_test_label_modified=np.zeros([250,10304])

    for i in range(50):

        non_faces_train_data_modified[i]=nonfaces_train_data[i]


        non_faces_test_data_modified[i]=nonfaces_test_data[i]


        non_faces_train_label_modified[i]=nonfaces_train_label[i]


        non_faces_test_label_modified[i]=nonfaces_test_label[i]


    for i in range(50,250,1):
        non_faces_train_data_modified[i]=nonfaces_train_data[i+150]
        non_faces_test_data_modified[i]=nonfaces_test_data[i+150]
        non_faces_train_label_modified[i]=nonfaces_train_label[i+150]
        non_faces_test_label_modified[i]=nonfaces_test_label[i+150]



    non_faces_train_data_modified =np.array(non_faces_train_data_modified)
    non_faces_test_data_modified = np.array(non_faces_test_data_modified)
    non_faces_train_label_modified =np.array(non_faces_train_label_modified)
    non_faces_test_label_modified =np.array(non_faces_test_label_modified)


    LDA_X_train_reduced, U_LDA = lda_large(non_faces_train_data_modified, non_faces_train_label_modified)
    print("data projected successfully ! \n")
    LDA_X_test_reduced = np.matmul(non_faces_test_data_modified, U_LDA)

    knn_lda = KNeighborsClassifier(n_neighbors=1)
    knn_lda.fit(LDA_X_train_reduced, non_faces_train_label_modified)
    labels_pred = knn_lda.predict(LDA_X_test_reduced)

    # # Compute the accuracy
    acc_lda = np.mean(np.array(labels_pred) == non_faces_test_label_modified)
    print(f"accuracy = {acc_lda}")




def pca_lda_variations(train_data, train_label,test_data, test_label):
    # Fit the LDA model
    print(train_data.shape)
    t0 = time()
    # apply Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(n_components=39)
    train_data = lda.fit_transform(train_data, train_label)
    test_data = lda.transform(test_data)

    # classify using random forest classifier
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(train_data, train_label)
    y_pred = classifier.predict(train_data)

    # print the accuracy and confusion matrix
    conf_m = confusion_matrix(test_label, y_pred)
    print(conf_m)
    print('Accuracy : ' + str(accuracy_score(test_label, y_pred)))

    t1 = time()

    print('LDA takes %f secs' % (t1 - t0))

    pca = PCA(n_components=39)
    X_train = pca.fit_transform(train_data)
    X_test = pca.transform(test_data)

    classifier = RandomForestClassifier()
    classifier.fit(train_data, train_label)

    # Predicting the Test set results
    y_pred = classifier.predict(test_data)

    cm = confusion_matrix(test_label, y_pred)
    print(cm)
    acc = accuracy_score(test_label, y_pred)
    print('Accuracy', acc)

    t2 = time()

    print('PCA takes %f secs' % (t2-t1))







def lda_tuner(train_data,train_label,test_data,test_label):
    # call LDA
    accuracies_lda = []
    k_values = [1, 3, 5, 7]
    LDA_X_train_reduced, U_LDA = lda_org(train_data, train_label)
    LDA_X_test_reduced = np.matmul(test_data, U_LDA)

    for i in range(1, 8, 2):
        knn_lda = KNeighborsClassifier(n_neighbors=i)
        knn_lda.fit(LDA_X_train_reduced, train_label)
        labels_pred = knn_lda.predict(LDA_X_test_reduced)
        acc_lda = np.mean(np.array(labels_pred) == test_label)
        accuracies_lda.append(acc_lda)
        print(f"accuracy = {acc_lda}")
    plt.plot(k_values, accuracies_lda, label='LDA')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('K-NN Classifier Performance')
    plt.legend()
    plt.show()


def pca_tuner(train_data,train_label,test_data,test_label):
    # call PCA
    alphas = [0.85]
    accuracies_pca = []
    k_values = [1, 3, 5, 7]
    for alpha in alphas:
        # Project the training set and test set using the same projection matrix
        pca_X_train_reduced, U_PCA = pca(train_data, alpha)
        pca_X_test_reduced = np.matmul(test_data, U_PCA)

        for i in range(1, 8, 2):
            knn_pca = KNeighborsClassifier(n_neighbors=i)
            knn_pca.fit(pca_X_train_reduced, train_label)
            labels_pred_pca = knn_pca.predict(pca_X_test_reduced)

            # # Compute the accuracy
            acc_pca = np.mean(np.array(labels_pred_pca) == test_label)
            accuracies_pca.append(acc_pca)
            print(f"accuracy = {acc_pca}")
    plt.plot(k_values, accuracies_pca, label='PCA')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('K-NN Classifier Performance')
    plt.legend()
    plt.show()






def main():
    print("Hello pattern recoganization ! \n")
    x_train_data,x_train_label,y_test_data,y_test_label,inp=generate_data()


    if inp==1:
        while(True):
            x=input("Enter 1 to run PCA\nEnter 2 to run LDA\nEnter 3 to run Classifier Tuning\nEnter 4 to run the bonus part\n")
            if x=="1":
                 print("RUNING PCA : ")
                 pca_call(x_train_data,x_train_label,y_test_data,y_test_label,inp)

            elif x=="2":
                print("RUNING LDA : ")
                lda_call(x_train_data,x_train_label,y_test_data,y_test_label,inp)

            elif x=="3":
                print("RUNING CLASSIFIER TUNING")
                pca_tuner(y_test_data,y_test_label,x_train_data,x_train_label)
                lda_tuner(y_test_data,y_test_label,x_train_data,x_train_label)

            elif x=="4":
                x = input("To run different training and tst splits press 1\nTo run other variations of PCA & LDA press 2:\n")
                if x == "1":
                    test_data, test_label, train_data, train_label = generate_bonus_data()
                    pca_call(train_data,train_label ,test_data, test_label,inp)
                elif x == "2":
                    pca_lda_variations(x_train_data, x_train_label, y_test_data, y_test_label)

                else:
                    return -1

            else:
                return -1




    elif inp ==2:
        while (True):
            x = input("Enter 1 to show Show failure and success cases.\nEnter 2 to show Plot the accuracy vs the number of non-face\nEnter 3 to show Criticize  the  accuracy  measure  for  large  numbers  of  non-faces\n")
            if x == "1":
                lda_call(x_train_data,x_train_label,y_test_data,y_test_label,inp)
            elif x == "2":
                lda_plot(x_train_data,x_train_label,y_test_data,y_test_label)
            elif x == "3":
                large_non_faces(x_train_data, x_train_label, y_test_data, y_test_label)
            else:
                return -1


main()


