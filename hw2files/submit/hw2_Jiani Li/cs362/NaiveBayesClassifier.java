package cs362;

import java.io.Serializable;
import java.util.List;
import java.util.Map.Entry;
import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayesClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	double lambda;
	HashMap<Integer, Double> labelOneFeatureProbability;
	HashMap<Integer, Double> labelNegOneFeatureProbability;
	double probabilityOneLabel;
	double probabilityNegOneLabel;
	Label NaiveBayesLabel;
	
	
	public NaiveBayesClassifier(Double lambda) {
		this.lambda = lambda;
	}
	
	public void train(List<Instance> instances){
		
		// get probability of +1 and -1 label		
		int oneLabelCount = 0;
		int negOneLabelCount = 0;
		Label oneLabel = new ClassificationLabel(1);
		Label negOneLabel = new ClassificationLabel(0);
		List<Instance> oneLabelInstances = new ArrayList<Instance>();
		List<Instance> negOneLabelInstances = new ArrayList<Instance>();
		
		for (Instance instance:instances) {
			Label label = instance.getLabel();
			if ((label.toString()).equals(oneLabel.toString())){
				oneLabelCount++;
			}else if ((label.toString()).equals(negOneLabel.toString())) {
				negOneLabelCount++;
			}
		}
		
		// smooth label
		probabilityOneLabel = (oneLabelCount + lambda)/(oneLabelCount + negOneLabelCount + 2.0 * lambda);
		probabilityNegOneLabel = (negOneLabelCount + lambda)/(oneLabelCount + negOneLabelCount + 2.0 * lambda);
		
		//System.out.println("one probabiltiy"+ probabilityOneLabel);
		//System.out.println("neg one probabiltiy"+ probabilityNegOneLabel);


		// handling non-binary features
		for (Instance instance: instances) {
			FeatureVector featureVector = instance.getFeatureVector();
			for (Entry<Integer, Double> feature: featureVector.featureVector.entrySet()) {
				if (feature.getValue()!=0 && feature.getValue()!= 1) {
					if (feature.getValue() < 0.5) {
						feature.setValue(0.0);
					}else {
						feature.setValue(1.0);
					}
				}
			}
		}
		
		// get conditional probability of features
		HashMap<Integer, Integer> labelOneFeatureCount = new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> labelNegOneFeatureCount = new HashMap<Integer, Integer>();
		for (Instance instance: instances) {
			FeatureVector featureVector = instance.getFeatureVector();
			for (Entry<Integer, Double> feature: featureVector.featureVector.entrySet()) {
				if ((instance.getLabel().toString()).equals(oneLabel.toString())) {
					if (feature.getValue() == 1) {
						labelOneFeatureCount.put(feature.getKey(), labelOneFeatureCount.get(feature.getKey()) == null? 1:labelOneFeatureCount.get(feature.getKey()) + 1);
					}
				}else if ((instance.getLabel().toString()).equals(negOneLabel.toString())) {
					if (feature.getValue() == 1) {
						labelNegOneFeatureCount.put(feature.getKey(), labelNegOneFeatureCount.get(feature.getKey()) == null? 1:labelNegOneFeatureCount.get(feature.getKey()) + 1);
					}
				}
			}
		}			
		labelOneFeatureProbability = new HashMap<Integer, Double>();
		labelNegOneFeatureProbability = new HashMap<Integer, Double>();
		
		int featureDimension = GetFeatureDimension(instances);		
		for (int i=1;i<=featureDimension;i++) {
			if (labelOneFeatureCount.containsKey(i)) {
				//smoothing
				double probability = (labelOneFeatureCount.get(i) + lambda)/(oneLabelCount + lambda * 2.0);
				labelOneFeatureProbability.put(i, probability);
			}else {
				//smoothing
				double probability = lambda/(oneLabelCount + lambda * 2.0);
				labelOneFeatureProbability.put(i, probability);
			}
		}
		
		for (int i=1;i<=featureDimension;i++) {
			if (labelNegOneFeatureCount.containsKey(i)) {
				//smoothing
				double probability = (labelNegOneFeatureCount.get(i) + lambda)/(negOneLabelCount + lambda * 2.0);
				labelNegOneFeatureProbability.put(i, probability);
			}else {
				//smoothing
				double probability = lambda/(negOneLabelCount + lambda * 2.0);
				labelNegOneFeatureProbability.put(i, probability);
			}
		}	
	}
		
	
	public Label predict(Instance instance) {
		//probability of label 1
		double logOneProb = Math.log(probabilityOneLabel);
		for (Entry<Integer, Double> feature: instance.getFeatureVector().featureVector.entrySet()) {
			if (labelOneFeatureProbability.get(feature.getKey()) != null) {
				logOneProb += Math.log(labelOneFeatureProbability.get(feature.getKey()));
			}
		}

		//probability of label -1
		double logNegOneProb = Math.log(probabilityNegOneLabel);
		for (Entry<Integer, Double> feature: instance.getFeatureVector().featureVector.entrySet()) {
			if (labelNegOneFeatureProbability.get(feature.getKey()) != null) {
				logNegOneProb += Math.log(labelNegOneFeatureProbability.get(feature.getKey()));
			}
		}
		
		if (logOneProb >= logNegOneProb) {
			NaiveBayesLabel = new ClassificationLabel(1);
		}else {
			NaiveBayesLabel = new ClassificationLabel(0);
		}
		
		//System.out.println("predict"+NaiveBayesLabel.toString().equals(instance.getLabel().toString()));
		
		return NaiveBayesLabel;	
	}
	
	public int GetFeatureDimension(List<Instance> instances) {
		int maxDimension = 0;
		for (Instance instance: instances) {
			FeatureVector featureVector = instance.getFeatureVector();
			int featureDimension = featureVector.getMaxIndex();
			if (featureDimension > maxDimension) {
				maxDimension = featureDimension;
			}
		}
		return maxDimension;
	}

}
