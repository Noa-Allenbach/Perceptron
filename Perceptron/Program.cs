using System;

public class Perceptron
{
    private double[] weights;
    private double learningRate;

    public Perceptron(int inputSize, double learningRate)
    {
        this.weights = new double[inputSize];
        this.learningRate = learningRate;

        // Initialize weights with random values
        Random random = new Random();
        for (int i = 0; i < inputSize; i++)
        {
            this.weights[i] = random.NextDouble() * 2 - 1;
        }
    }

    public int Predict(double[] inputs)
    {
        if (inputs.Length != weights.Length)
        {
            throw new ArgumentException("Input size does not match weight size");
        }

        double sum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * weights[i];
        }

        // Apply step function as activation
        int output = sum > 0 ? 1 : 0;
        return output;
    }

    public void Train(double[][] trainingData, int[] labels, int epochs)
    {
        if (trainingData.Length != labels.Length)
        {
            throw new ArgumentException("Number of training data points does not match number of labels");
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < trainingData.Length; i++)
            {
                double[] inputs = trainingData[i];
                int label = labels[i];

                int prediction = Predict(inputs);
                int error = label - prediction;

                // Update weights based on error and learning rate
                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * inputs[j];
                }
            }
        }
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        // Training data and labels for OR gate
        double[][] trainingData = {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };

        int[] labels = { 0, 1, 1, 1 };

        // Create a Perceptron with 2 input neurons and learning rate of 0.1
        Perceptron perceptron = new Perceptron(2, 0.1);

        // Train the perceptron
        perceptron.Train(trainingData, labels, epochs: 100);

        // Test the perceptron
        Console.WriteLine("Testing perceptron:");
        Console.WriteLine($"0 OR 0 = {perceptron.Predict(new double[] { 0, 0 })}");
        Console.WriteLine($"0 OR 1 = {perceptron.Predict(new double[] { 0, 1 })}");
        Console.WriteLine($"1 OR 0 = {perceptron.Predict(new double[] { 1, 0 })}");
        Console.WriteLine($"1 OR 1 = {perceptron.Predict(new double[] { 1, 1 })}");
        Console.WriteLine($"1 OR 1 = {perceptron.Predict(new double[] { 1, 2 })}");

        Console.ReadLine();
    }
}