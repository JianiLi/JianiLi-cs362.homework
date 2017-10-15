package cs362;

import java.io.Serializable;
import java.util.List;
import java.util.TreeMap;
import java.util.Map.Entry;

public class PerceptronPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	Double online_learning_rate;
	Integer online_training_iterations;
	Label PerceptronLabel;
	TreeMap<Integer, Double> w;

	public PerceptronPredictor(Double online_learning_rate, Integer online_training_iterations) {
		this.online_learning_rate = online_learning_rate;
		this.online_training_iterations = online_training_iterations;
	}
	
	public void train(List<Instance> instances){
		
		Integer wDimension = GetWeightDimension(instances);
		// System.out.println("w dimension"+wDimension);
		w = new TreeMap<Integer, Double>();
		double wInitValue = 0;
		for (int i = 1; i <= wDimension; i++) {
			w.put(i, wInitValue);
		}
		
		Label predictedLabel;
		
		for (Instance instance: instances) {
			Label label = instance.getLabel();
			Integer labelInt = Integer.parseInt(label.toString());
			if (labelInt == 0) {
				Label newLabel = new ClassificationLabel(-1);
				instance._label = newLabel;
				//System.out.println("new label"+instance.getLabel());
			}
		}

		for (int k = 1; k <= this.online_training_iterations; k++) {
			for (Instance instance:instances) {
				FeatureVector featureVector = instance.getFeatureVector();
				double dotProductResults = dotProduct(w, featureVector);
				//System.out.println("dot product"+dotProductResults);
				if (dotProductResults >= 0) {
					predictedLabel = new ClassificationLabel(1);
				}else {
					predictedLabel = new ClassificationLabel(-1);
				}
				if (!predictedLabel.toString().equals(instance.getLabel().toString())) {
					updateW(instance.getLabel(), featureVector);
				}
			}
		}
		
		for (Instance instance: instances) {
			Label label = instance.getLabel();
			Integer labelInt = Integer.parseInt(label.toString());
			if (labelInt == -1) {
				Label newLabel = new ClassificationLabel(0);
				instance._label = newLabel;
				//System.out.println("new label"+instance.getLabel());
			}
		}
		/*for (Entry<Integer,Double> entry: w.entrySet()) {
			System.out.print(entry.getValue());
		}*/
	}	
	
	public Label predict(Instance instance) {
		FeatureVector featureVector = instance.getFeatureVector();
		double dotProductResults = dotProduct(w,featureVector);
		if (dotProductResults >= 0) {
			PerceptronLabel = new ClassificationLabel(1);
		}else {
			PerceptronLabel = new ClassificationLabel(0);
		}		
		return PerceptronLabel;	
	}

	public void updateW(Label trueLabel, FeatureVector featureVector) {
		for (Entry<Integer, Double> feature: featureVector.featureVector.entrySet()) {
			int labelValue = Integer.parseInt(trueLabel.toString());
			double newW = w.get(feature.getKey()) + this.online_learning_rate * labelValue * feature.getValue();
			w.put(feature.getKey(), newW);
		}
	}
	
	public double dotProduct(TreeMap<Integer, Double> w, FeatureVector featureVector) {
		double results = 0;
		for (Entry<Integer, Double> feature: featureVector.featureVector.entrySet()) {
			if (w.get(feature.getKey()) != null) {
				results += w.get(feature.getKey()) * feature.getValue();
			}
		}
		return results;
	}
	
	public int GetWeightDimension(List<Instance> instances) {
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
