import numpy as np
import math

class Perceptron:
    def __init__(self, input_size, learning_rate=0.5, lamb=0.1):
        self.weights = np.random.rand(input_size)
        self.learning_rate = learning_rate
        self.lamb = lamb

    def activation(self, u):
        return 2 / (1 + math.exp(-u)) - 1

    def predict(self, inputs):
        u = np.dot(self.weights, inputs)
        return self.activation(self.lamb * u)

    def train(self, training_data, targets, error_threshold=0.01, max_iterations=1000):
        for epoch in range(max_iterations):
            total_error = 0
            
            for inputs, target in zip(training_data, targets):
                output = self.predict(inputs)
                
                error = target - output
                total_error += 0.5 * error ** 2
                
                delta = error * (1 - output ** 2) / 2
                self.weights += self.learning_rate * delta * inputs

            print(f"Época {epoch + 1} | Erro Total: {total_error:.4f} | Pesos: {self.weights}")

            if total_error < error_threshold:
                print(f"Convergência atingida após {epoch + 1} épocas.")
                break
        else:
            print("Número máximo de iterações atingido sem convergência.")
    
    def test(self, test_data):
        predictions = []
        for inputs in test_data:
            output = self.predict(inputs)
            predictions.append(output)
        return predictions

training_inputs = np.array([[1, 1, -1],
                            [1, -1, -1],
                            [-1, 1, -1],
                            [-1, -1, -1]])

targets = np.array([1, 1, 1, -1])

perceptron = Perceptron(input_size=3)

perceptron.train(training_inputs, targets)

outputs = perceptron.test(training_inputs)

for input_data, output in zip(training_inputs, outputs):
    print(f"Entrada: {input_data} | Saída prevista: {output:.4f}")
