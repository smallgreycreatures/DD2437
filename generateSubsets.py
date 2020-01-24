def generateSubsets((n,scenario,shuffle=1, verbose=1, pattern=1):
    #50% each
    if scenario==1:
        class_A = np.concatenate((np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_A) +  mu_A[0],np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_A) +  mu_A[1]))
        class_B = np.concatenate((np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_B) +  mu_B[0],np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_B) +  mu_A[1]))

    #50% of A, 100% of B
    if scenario==2:
        class_A = np.concatenate((np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_A) +  mu_A[0],np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_A) +  mu_A[1]))
        class_B = np.concatenate((np.multiply(np.random.normal(0,1,(1,n)),sigma_B) +  mu_B[0],np.multiply(np.random.normal(0,1,(1,n)),sigma_B) +  mu_A[1]))

    #50% of B, 100% of A
    if scenario==3:
        class_A = np.concatenate((np.multiply(np.random.normal(0,1,(1,int(n))),sigma_A) +  mu_A[0],np.multiply(np.random.normal(0,1,(1,int(n))),sigma_A) +  mu_A[1]))
        class_B = np.concatenate((np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_B) +  mu_B[0],np.multiply(np.random.normal(0,1,(1,int(n/2))),sigma_B) +  mu_A[1]))

    #20% 80% and stuff
    if scenario==4:
        class_A = np.concatenate((np.multiply(np.random.normal(0,1,(1,n)),sigma_A) +  mu_A[0],np.multiply(np.random.normal(0,1,(1,n)),sigma_A) +  mu_A[1]))
        class_B = np.concatenate((np.multiply(np.random.normal(0,1,(1,n)),sigma_B) +  mu_B[0],np.multiply(np.random.normal(0,1,(1,n)),sigma_B) +  mu_A[1]))
        c_a = np.zeros((2,20))
        c_b = np.zeros((2,80))
        #print(class_A.shape)
        a_i = 0
        b_i = 0
        for i in range(n):
            if class_A[1,i] < 0 and c_a[1,19] == 0:
                c_a[:,a_i] = class_A[:,i]
                a_i+= 1
            elif class_A[1,i] >0 and c_b[1,79] == 0:
                c_b[:,b_i] = class_A[:,i]
                b_i+=1
        class_A = c_a
        class_B = c_b

    if shuffle:
        patterns=np.concatenate((class_A,class_B),axis=1)
        patterns = np.concatenate((patterns,np.ones((1,patterns.shape[1])))
        targets=np.concatenate((np.ones(n),-np.ones(n)))
        i = np.arange(patterns.shape[1])
        np.random.shuffle(i)
        patterns=patterns[:,i]
        targets=targets[i]
        #print(patterns)
        #print(targets)
    else:
        patterns = np.concatenate((class_A,class_B),axis=1)
        targets = np.array([np.ones(n,),-np.ones(n,)])
        targets = np.ravel(targets)

    if verbose:
        #Plotting classes if desired
        plt_A, = plt.plot(class_A[0,:],class_A[1,:],'g^', label='Class A')
        plt_B, = plt.plot(class_B[0,:],class_B[1,:],'bs', label='Class B')
        plt.legend(handles=[plt_A, plt_B])
        plt.show()



    if pattern:
        return patterns, targets
    else:
        return class_A, class_B
