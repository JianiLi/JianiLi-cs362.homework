package cs362;

import java.util.Map.Entry;

public class PolynomialKernelLogisticRegression extends KernelLogisticRegression{
	
	private double d;
	
    public PolynomialKernelLogisticRegression(Double learning_rate, Integer training_iterations, double d) {
        super(learning_rate, training_iterations);
        this.d = d;
    }
    
    public double computeKernel(FeatureVector x, FeatureVector z) {
		double dotProduct = 0;
		for (Entry<Integer, Double> x_feature: x.featureVector.entrySet()) {
			int index = x_feature.getKey();
			if (z.featureVector.containsKey(index)) {
				dotProduct += x_feature.getValue() + z.featureVector.get(index);
			}
		}
		
		double kernel = Math.pow(dotProduct + 1.0, d);
		return kernel;
    }
}
