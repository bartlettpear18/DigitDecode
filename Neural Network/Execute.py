import Network
import Data_Prep

training_data, validation_data, test_data = Data_Prep.load_data_wrapper()
training_data = list(training_data)

neuralNet = Network.Network([784, 30, 10])
neuralNet.stochasticGradientAlgorithm(training_data, 30, 10, 3.0, test_data = test_data)
