import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class LogisticRegressionSGD {
    private double learningRate;
    private int iterations;
    private double[] weights;

    public LogisticRegressionSGD(double learningRate, int iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.weights = null;
    }

    public static List<String[]> loadCsv(String filepath) {
        List<String[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                data.add(values);
            }
            System.out.println("Loaded " + (data.size() - 1) + " rows from " + filepath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        data.remove(0); // Remove header
        return data;
    }

    public static double dotProduct(double[] v1, double[] v2) {
        double sum = 0;
        for (int i = 0; i < v1.length; i++) {
            sum += v1[i] * v2[i];
        }
        return sum;
    }

    public static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public void fit(double[][] X, int[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        this.weights = new double[nFeatures];

        for (int iteration = 0; iteration < this.iterations; iteration++) {
            for (int i = 0; i < nSamples; i++) {
                double linearCombination = dotProduct(X[i], this.weights);
                double yPredicted = sigmoid(linearCombination);
                for (int j = 0; j < nFeatures; j++) {
                    this.weights[j] += this.learningRate * (y[i] - yPredicted) * X[i][j];
                }
            }
            if ((iteration + 1) % 10 == 0) {
                System.out.println("Iteration " + (iteration + 1) + "/" + this.iterations);
            }
        }
    }

    public double[] predictProb(double[][] X) {
        double[] probabilities = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = sigmoid(dotProduct(X[i], this.weights));
        }
        return probabilities;
    }

    public int[] predict(double[][] X) {
        double[] probabilities = predictProb(X);
        int[] predictions = new int[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            predictions[i] = probabilities[i] >= 0.5 ? 1 : 0;
        }
        System.out.println("Generated predictions for " + predictions.length + " samples");
        return predictions;
    }

    public static double[][] preprocessFeatures(List<String[]> data) {
        double[][] features = new double[data.size()][data.get(0).length - 1];
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.get(i).length - 1; j++) {
                features[i][j] = Double.parseDouble(data.get(i)[j]);
            }
        }
        return features;
    }

    public static int[] preprocessLabels(List<String[]> data) {
        int[] labels = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = Integer.parseInt(data.get(i)[data.get(i).length - 1]);
        }
        return labels;
    }

    public static void evaluateMetrics(int[] yTrue, int[] yPred) {
        int truePositive = 0;
        int trueNegative = 0;
        int falsePositive = 0;
        int falseNegative = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1 && yPred[i] == 1) truePositive++;
            else if (yTrue[i] == 0 && yPred[i] == 0) trueNegative++;
            else if (yTrue[i] == 0 && yPred[i] == 1) falsePositive++;
            else if (yTrue[i] == 1 && yPred[i] == 0) falseNegative++;
        }
        double accuracy = (double) (truePositive + trueNegative) / (yTrue.length);
        double precision = truePositive / (double) (truePositive + falsePositive);
        double recall = truePositive / (double) (truePositive + falseNegative);
        double f1Score = 2 * (precision * recall) / (precision + recall);

        System.out.printf("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1 Score: %.4f%n", accuracy, precision, recall, f1Score);
    }

    public static void main(String[] args) {
        List<String[]> trainData = loadCsv("./train-1.csv");
        List<String[]> testData = loadCsv("./test-1.csv");

        double[][] XTrain = preprocessFeatures(trainData);
        int[] yTrain = preprocessLabels(trainData);
        double[][] XTest = preprocessFeatures(testData);
        int[] yTest = preprocessLabels(testData);

        LogisticRegressionSGD model = new LogisticRegressionSGD(0.01, 200);
        System.out.println("Starting training...");
        model.fit(XTrain, yTrain);

        System.out.println("Predicting on test set...");
        int[] predictions = model.predict(XTest);

        evaluateMetrics(yTest, predictions);
    }
}

//Accuracy: 0.9561, Precision: 0.8111, Recall: 0.9068, F1 Score: 0.8563