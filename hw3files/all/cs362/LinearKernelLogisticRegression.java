package cs362;

import java.util.Map.Entry;

public class LinearKernelLogisticRegression extends KernelLogisticRegression{
	
    public LinearKernelLogisticRegression(Double learning_rate, Integer training_iterations) {
        super(learning_rate, training_iterations);
    }
    
    public double computeKernel(FeatureVector x, FeatureVector z) {
    		double kernel = 0;
    		for (Entry<Integer, Double> x_feature: x.featureVector.entrySet()) {
    			int index = x_feature.getKey();
    			if (z.featureVector.containsKey(index)) {
    				kernel += x_feature.getValue() * z.featureVector.get(index);
    			}
    		}
	
    		return kernel;
    }

}
