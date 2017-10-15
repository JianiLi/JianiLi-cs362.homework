package cs362;

import java.io.Serializable;
import java.util.HashMap;

public class FeatureVector implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	HashMap<Integer, Double> featureVector = new HashMap<Integer, Double>();

	public void add(int index, double value) {
		// TODO Auto-generated method stub
		featureVector.put(index, value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		if (featureVector.get(index) == null) {
			return 0;
		}else {
			double value = featureVector.get(index);
			return value;
		}
	}
}
