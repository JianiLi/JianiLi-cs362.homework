package cs362;

import java.util.List;

public class Evaluator {
	
	int correctPredictLabelNum = 0;
	int predictLabelNum = 0;

	public double evaluate(List<Instance> instances, Predictor predictor) {
		
		if (predictor == null) {
			System.out.println("No predictor");
			return 0;
		}else if (instances.size() == 0) {
			System.out.println("No instances");
			return 0;
		}		
		
		for (Instance instance: instances) {
			Label predictLabel = predictor.predict(instance);
			Label actualLabel = instance.getLabel();
			if (predictLabel != null && actualLabel != null) {
				if (predictLabel.toString().equals(actualLabel.toString())) {
					correctPredictLabelNum++;
				}
				predictLabelNum++;
			}		
		}
		
		if(predictLabelNum != 0) {
			double accuracy = (float)correctPredictLabelNum/predictLabelNum;
			return accuracy;
		}else {
			System.out.println("Labels are not available");
			return 0;
		}
	}
}
