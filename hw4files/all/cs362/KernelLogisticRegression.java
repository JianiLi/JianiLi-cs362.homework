package cs362;

import java.io.Serializable;
import java.util.List;

public abstract class KernelLogisticRegression extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	Double learning_rate;
	Integer training_iterations;
	private List<Instance> trainingInstances;
	private double[] alpha;
	

	public KernelLogisticRegression(Double learning_rate, Integer training_iterations) {
		this.learning_rate = learning_rate;
		this.training_iterations = training_iterations;
	}
	
	public abstract double computeKernel(FeatureVector x, FeatureVector z);
	
	public double logisticFunction(double z) {
		double g = 1.0 / (1 + Math.exp(-1*z));
		return g;
	}
	
	public double ComputeTrainingSetKernel(FeatureVector x, FeatureVector z, int xIndex, int zIndex, Double[][] gramMatrix) {
		int row = xIndex;
		int col = zIndex;
		if (xIndex > zIndex) {
			row = zIndex;
			col = xIndex;
		}
		if (gramMatrix[row][col] == null) {
			gramMatrix[row][col] = computeKernel(x,z);
		}
		
		return gramMatrix[row][col];
	}
	
	public void train(List<Instance> instances){
		trainingInstances = instances;
		int instancesNum = instances.size();
		Double[][] gramMatrix = new Double[instancesNum][instancesNum];
		alpha = new double[instancesNum];
		
		for(int iter = 0; iter < training_iterations; iter++) {
			double[] xKernel = new double[instancesNum];
			double[] updatedAlpha = new double[instancesNum];
			for(int i = 0; i < instancesNum; i++) {
				double xiKernel = 0;
				Instance xi = instances.get(i);
				for(int j = 0; j < instancesNum; j++) {
					Instance xj = instances.get(j);
					xiKernel += alpha[j] * ComputeTrainingSetKernel(xj.getFeatureVector(), xi.getFeatureVector(), j,i,gramMatrix);
				}
				xKernel[i] = xiKernel;
			}
			
			for(int k = 0; k < instancesNum; k++) {
				double partialDerivative = 0;
				Instance xk = instances.get(k);
				for(int i = 0; i < instancesNum; i++) {
					Instance xi = instances.get(i);
					ClassificationLabel yi = (ClassificationLabel) xi.getLabel();
					double xiKernel = xKernel[i];
					double xixkKernel = ComputeTrainingSetKernel(xi.getFeatureVector(), xk.getFeatureVector(), i, k, gramMatrix);
					ClassificationLabel oneLabel = new ClassificationLabel(1);
					ClassificationLabel zeroLabel = new ClassificationLabel(0);
					if(yi.toString().equals(oneLabel.toString())) {
						xiKernel = -1.0 * xiKernel;
					}else if(yi.toString().equals(zeroLabel.toString())) {
						xixkKernel = -1.0 * xixkKernel;
					}
					partialDerivative += logisticFunction(xiKernel) * xixkKernel;
				}
				updatedAlpha[k] = alpha[k] + learning_rate * partialDerivative;
			}
			alpha = updatedAlpha;
		}
		
	}
	
	public Label predict(Instance instance) {
		double sum = 0;
		for(int j = 0; j < trainingInstances.size(); j++) {
			double xjxiKernel = computeKernel(trainingInstances.get(j).getFeatureVector(), instance.getFeatureVector());
			sum += alpha[j] * xjxiKernel;
		}
		if(logisticFunction(sum) >= 0.5) {
			return new ClassificationLabel(1);
		}else {
			return new ClassificationLabel(0);
		}
	}

}
