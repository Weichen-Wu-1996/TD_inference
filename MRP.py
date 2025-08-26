import numpy as np
from scipy.linalg import eig

class Markov_Reward_Process:
    def __init__(self,
                 S: int,
                 d: int,
                 gamma: float,
                 P: np.array = None,
                 r: np.array = None,
                 Phi: np.array = None,
                 default_setup: bool = True,
                 ):
        '''
        Initialization of the Markov Reward Process
        S: Cardinality of the state space
        d: feature dimension (d <= S)
        gamma: discount factor, 0 <= gamma < 1
        P: transition matrix, dim(P) = (S,S)
        r: reward function, dim(r) = S
        Phi: feature matrix, dim(Phi) = (S,d)
        default_setup: flag to indicate whether to use the default setup
        '''
        self.S = S
        assert d <= S, "d must not be larger than S!"
        self.d = d
        assert gamma >= 0 and gamma < 1, "gamma must be in [0,1)!"
        self.gamma = gamma

        if default_setup:
            self.default_setup()
        else:
            assert P.shape == (S,S), "Wrong shape for transition matrix!"
            self.P = P
            assert r.shape == (S,), "Wrong shape for reward function!"
            self.r = r
            assert Phi.shape == (S,d), "Wrong shape for features!"
            self.Phi = Phi

        self.get_params()

    def default_setup(self):
        '''
        Default setup for the MRP
        Details in Appendix B of https://arxiv.org/pdf/2305.19001
        '''
        self.r = np.ones((self.S,))
        self.r[0:(self.d-1)] = 0
        
        epsl = 0.01
        q = (self.gamma - (1-self.gamma) * (1-self.gamma) * epsl) * np.ones((self.d-1,))
        for i in range(int((self.d-1)/2)):
            q[i] = q[i] + 2 * (1-self.gamma) * (1-self.gamma) * epsl
        p = (1-q) / ((1-self.gamma) * (self.d-1))
        self.P = np.zeros((self.S,self.S))
        for i in range(self.d-1):
            self.P[i,i] =  q[i]
            for j in range(self.d-1,self.S):
                self.P[i,j] = (1-q[i])/(self.S-self.d+1)
        for i in range(self.d-1,self.S):
            for j in range(self.d-1):
                self.P[i,j] = (1-self.gamma) * p[j]
            for j in range(self.d-1,self.S):
                self.P[i,j] = self.gamma / (self.S-self.d+1)

        self.Phi = np.zeros((self.S,self.d))
        for i in range(self.S):
            self.Phi[i,min(i,self.d-1)] = 1

    def get_params(self):
        '''
        Function to get the parameters of the MRP
        '''
        # Value function
        I = np.identity(self.S)
        self.V = np.linalg.solve((I - self.gamma * self.P), self.r)

        # Stationary distribution
        eigvals, eigvecs = eig(self.P.T)
        mu = np.real(eigvecs[:, np.isclose(eigvals, 1)])
        self.mu = mu[:, 0] / mu[:, 0].sum()   # normalize

        # Matrix A and vector b for TD learning
        A = np.dot(np.transpose(self.Phi),((I - self.gamma * self.P) * self.mu.reshape((-1,1))))
        self.A = np.dot(A,self.Phi)
        self.b = np.dot(np.transpose(self.Phi),(self.mu * self.r).reshape((-1,1)))

        # Feature covariance matrix
        self.Sigma = np.dot(np.transpose(self.Phi), self.mu.reshape((self.S,1)) * self.Phi)

        # Optimal coefficients, target of TD learning
        self.theta_star = np.linalg.solve(self.A,self.b)

        # Covariance of TD error
        self.Gamma = np.zeros((self.d,self.d))
        for s1 in range(self.S):
            for s2 in range(self.S):
                TD_err = self.r[s1] - np.dot((self.Phi[s1] - self.gamma * self.Phi[s2]),self.theta_star)
                Ct = TD_err * self.Phi[s1]
                self.Gamma += Ct * Ct.reshape((-1,1)) * self.mu[s1] * self.P[s1,s2]

        # Asymptotic covariance of averaged TD estimator
        self.A_inv = np.linalg.inv(self.A)
        self.Lambda_star = np.dot(np.dot(self.A_inv,self.Gamma),np.transpose(self.A_inv))