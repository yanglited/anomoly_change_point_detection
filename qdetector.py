import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
np.set_printoptions(precision=2, suppress=True, threshold=10000, edgeitems=10, linewidth=1000)

class Qdetector(object):

    def __init__(self, threshold, data):
        self.threshold = threshold
        self.data = data
        try:
            dim, length = self.data.shape
        except ValueError:
            self.data = np.expand_dims(self.data, axis = 0)
            dim, length = self.data.shape

        self.dimensionality = dim
        self.numSteps = length

        self.detectPoint = None
        self.changePoint = None
        self.DMat      = np.zeros((self.dimensionality, self.numSteps, self.numSteps))
        self.Rn_of_Zi  = np.zeros((self.dimensionality, self.numSteps))
        self.Rn_bar_nk = np.zeros((self.dimensionality, self.numSteps))
        self.Rkn       = np.zeros((1, self.numSteps))
        self.Rkn_max = None
        self.detectionFlag = False

    def detect(self):
        Z1toN = self.data.copy()
        n = self.numSteps
        for i in np.arange(n):
            for j in np.arange(n):
                if i !=j:
                    self.DMat[:,i,j] = ( Z1toN[:,i] - Z1toN[:,j] )/np.linalg.norm(Z1toN[:,i] - Z1toN[:,j])

        self.Rn_of_Zi = np.sum(self.DMat, axis=2)

        for k in np.arange(n):
            self.Rn_bar_nk[:,k] = (1.0/(k+1))*np.sum(self.Rn_of_Zi[:, 0:k+1], axis=1)

        Cov_Mat_Est = np.zeros((self.dimensionality, self.dimensionality, n));
        for k in np.arange(n):#assuming the last data point cannot be detected as a change point, since covariance matrix will become a zero matrix.
            Cov_Mat_Est[:,:,k] = (n-(k+1))/((n-1.0)*n*(k+1))*np.matmul(self.Rn_of_Zi, self.Rn_of_Zi.transpose())
            print(Cov_Mat_Est[:,:,k])
            inv_Cov = np.linalg.inv(Cov_Mat_Est[:,:,k]+0.000001*np.eye(self.dimensionality, dtype=np.float64))
            self.Rkn[0,k] = np.matmul( np.matmul(self.Rn_bar_nk[:,k].transpose(), inv_Cov), self.Rn_bar_nk[:,k] )

        self.changePoint = np.argmax(self.Rkn)
        if self.Rkn[0, self.changePoint] >= self.threshold:
            self.detectionFlag = True
            print("Change point detected at sample # %d.", self.changePoint)

def main():
    numOfEvents = 2
    stateVector = np.asarray([1,1,0,1,0,0,1,1,1,0,1,1,1,1,0,0])
    stateVector = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    stateVector = np.asarray([1,0])
    # stateVector = np.random.randint(2, size=numOfEvents)
    sojournTimeVector = np.random.randint(1, high=300, size=numOfEvents)
    groundTruth = np.repeat(stateVector, sojournTimeVector)
    numberOfSamples = np.sum(sojournTimeVector)

    sigma = .5
    np.random.seed()
    samples1 = sigma * np.random.randn(groundTruth.size) + groundTruth
    samples2 = sigma * np.random.randn(groundTruth.size) + groundTruth

    samples = np.vstack((samples1, samples2))
    # samples = samples1
    threshold = numberOfSamples/3.0
    detector = Qdetector(30, samples)
    detector.detect()

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(groundTruth.T, '-o')
    ax1.plot(detector.changePoint, groundTruth[detector.changePoint], '-ro')
    ax2.plot(samples.T, '-o')
    ax3.plot(detector.Rkn.T, '-o')
    ax3.plot(detector.changePoint, detector.Rkn[0, detector.changePoint], '-ro')
    ax1.set(xlabel='sequence number', ylabel='State value',)
    ax2.set(xlabel='sequence number', ylabel='Sample value',)
    ax3.set(xlabel='sequence number', ylabel='Test statistic Rkn')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()

if __name__ == "__main__":
	main()
