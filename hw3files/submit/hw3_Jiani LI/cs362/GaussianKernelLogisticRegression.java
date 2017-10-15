package cs362;

import java.util.Map.Entry;

public class GaussianKernelLogisticRegression extends KernelLogisticRegression{
	
	private double sigma;
	
    public GaussianKernelLogisticRegression(Double learning_rate, Integer training_iterations, double sigma) {
        super(learning_rate, training_iterations);
        this.sigma = sigma;
    }
    
    public double computeKernel(FeatureVector x, FeatureVector z) {
		double norm = 0;
		for (Entry<Integer, Double> x_feature: x.featureVector.entrySet()) {
			int index = x_feature.getKey();
			if (z.featureVector.containsKey(index)) {
				norm += Math.pow((x_feature.getValue() - z.featureVector.get(index)), 2.0);
			}else {
				norm += Math.pow(x_feature.getValue(), 2.0);
			}
		}
		
		for (Entry<Integer, Double> z_feature: z.featureVector.entrySet()) {
			int index = z_feature.getKey();
			if (!x.featureVector.containsKey(index)) {
				norm += Math.pow(z.featureVector.get(index), 2.0);
			}
		}
		
		double kernel = Math.exp(-1.0*norm/(2*sigma*sigma));
		return kernel;
    }
}
