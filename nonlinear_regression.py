import numpy as np

np.random.seed(0)

class NLR:
    
    def __init__(self, dotp):
        """
        Constructor of the NLR class.
        
        Parameters:
        dotp -- the degree of the polynomial used to approximate the data.
        """
        self.dotp = dotp
    
    def change_matrix(self, inputs):
        """
        Function to create a data matrix augmented by polynomial degrees.
        
        Parameters:
        inputs -- input data matrix.
        
        Returns:
        A -- new data matrix augmented by polynomial degrees.
        """
        n_samples, _ = inputs.shape
        a_ = np.concatenate((np.ones([n_samples, 1]), inputs), axis=1)
        A = [a_]
        for i in range(1, self.dotp):
            A.append(inputs ** (i + 1))
        return np.concatenate(A, axis=1)
    
    def fit(self, X, y):
        """
        Function to train the model with input data.
        
        Parameters:
        X -- input data matrix.
        y -- output vector.
        
        Returns:
        self -- the NLR class itself.
        """
        self.matrix = self.change_matrix(X)
        self.w = np.linalg.pinv(self.matrix) @ y
        return self
    
    def predict(self, inputs):
        """
        Function to predict output with input data.
        
        Parameters:
        inputs -- input data matrix.
        
        Returns:
        outputs -- predicted output vector.
        """
        inputs = self.change_matrix(inputs)
        outputs = inputs @ self.w
        return outputs
