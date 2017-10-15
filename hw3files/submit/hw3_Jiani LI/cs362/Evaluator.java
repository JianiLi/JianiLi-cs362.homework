package cs362;

import java.util.List;

public class Evaluator {

	public double classsificationEvaluate(List<Instance> instances, Predictor predictor) {
		
		int correctPredictLabelNum = 0;
		int predictLabelNum = 0;
		
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
			double accuracy = correctPredictLabelNum * 1.0 /predictLabelNum;
			return accuracy;
		}else {
			System.out.println("Labels are not available");
			return 0;
		}
	}
	
	public double regressionEvaluate(List<Instance> instances, Predictor predictor) {
		
		if (predictor == null) {
			System.out.println("No predictor");
			return 0;
		}else if (instances.size() == 0) {
			System.out.println("No instances");
			return 0;
		}		
		
		double totalError = 0;
		int predictLabelNum = 0;
		
		for (Instance instance: instances) {
			Label predictLabel = predictor.predict(instance);
			double predictLabelValue = Double.parseDouble(predictLabel.toString());
			Label actualLabel = instance.getLabel();
			double actualLabelValue = Double.parseDouble(actualLabel.toString());
			if (predictLabel != null && actualLabel != null) {
				totalError += Math.abs(predictLabelValue - actualLabelValue);
				predictLabelNum++;
			}		
		}
		
		if(predictLabelNum != 0) {
			double accuracy = totalError/predictLabelNum;
			return accuracy;
		}else {
			System.out.println("Labels are not available");
			return 0;
		}
	}
}
