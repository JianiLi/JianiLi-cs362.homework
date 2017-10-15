package cs362;

import java.io.Serializable;
import java.util.Random;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections; 
import org.apache.commons.math3.linear;

public class LinearRegressionPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;

	Label linearRegressionLabel;
	
	public void train(List<Instance> instances){
		List X = new ArrayList();
		List<Label> y = new ArrayList<>();
		
		for (Instance instance : instances){
			FeatureVector featureVector = (FeatureVector)instance.getFeatureVector();
			featureVector.add(0,1);
			List<Integer> keyList = new ArrayList<>(featureVector.featureVector.keySet());
			Collections.sort(keyList);
			List<Double> x = new ArrayList<Double>();
			for (int i = 0; i < keyList.size(); i++) {
				x.add(featureVector.get(i));
			}
			X.add(x);
			Label label = instance.getLabel();
			y.add(label);
		}

		/*RealMatrix XMatrix = new Array2DRowRealMatrix(X);
		RealMatrix yMatrix = new Array2DRowRealMatrix(y);
		RealMatrix w = new RealMatrix();
		w = (((XMatrix.transpose().multiply(XMatrix)).getInverse()).multiply(XMatrix)).multiply(yMatrix);*/
		}
		
	
	public Label predict(Instance instance) {
	/*	ArrayList<double> x = new ArrayList<double>();
		FeatureVector featureVector = instance.getFeatureVector();
		featureVector.add(0,1);
		x.add(featureVector.values().toArray());
		linearRegressionLabel = (w.getInverse()).multiply(x);*/	
		return linearRegressionLabel;
		
	}
}
