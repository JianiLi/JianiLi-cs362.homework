package cs362;

import java.io.Serializable;
import java.util.List;

public class EvenOddPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	Label evenOddLabel;
	
	public void train(List<Instance> instances){
		// No training phase
	}
	
	public Label predict(Instance instance){
		double evenSum = 0.0;
		double oddSum = 0.0;
		
		FeatureVector featureVector = instance.getFeatureVector();

		for (Integer index : featureVector.featureVector.keySet()) {
			if (index % 2 == 0) {
				evenSum += featureVector.get(index);
			}else {
				oddSum += featureVector.get(index);
			}
		}
		
		if(evenSum >= oddSum){
			evenOddLabel = new ClassificationLabel(1);
		}else{
			evenOddLabel = new ClassificationLabel(0);
		}
		return evenOddLabel;
	}
}