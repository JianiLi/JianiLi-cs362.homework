package cs362;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map.Entry;

public class FeatureVector implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	HashMap<Integer, Double> featureVector = new HashMap<Integer, Double>();
	int cluster = 0;

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
	
	public int getMaxIndex() {
		int maxIndex = 0;
		for (Entry<Integer, Double> feature: featureVector.entrySet()) {
			if (feature.getKey() > maxIndex) {
				maxIndex = feature.getKey();
			}
		}
		return maxIndex;
	}
		
}
