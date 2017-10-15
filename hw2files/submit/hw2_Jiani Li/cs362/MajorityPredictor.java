package cs362;

import java.io.Serializable;
import java.util.Random;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

public class MajorityPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;

	Label majorityLabel;

	public void train(List<Instance> instances){
		HashMap<Label, Integer> labelList = new HashMap<Label, Integer>();
		for (Instance instance : instances){
			Label label = (Label) instance.getLabel();
			if(labelList.containsKey(label)){
				labelList.put(label,labelList.get(label) + 1);
			}else{
				labelList.put(label,1);
			}
		}
		
		int maxNum = 0;

		List<Label> maxLabels = new ArrayList<Label>();

		for (Label label : labelList.keySet()){
			if (labelList.get(label) >= maxNum){
				if (labelList.get(label) > maxNum){
					maxLabels.clear();
					maxNum = labelList.get(label);
					maxLabels.add(label);
					
				}else{
					maxLabels.add(label);
				}
			}
		}
		
		if (maxLabels.size() > 1){		
			Random randomGenerator = new Random();
			int index = randomGenerator.nextInt(maxLabels.size());
			majorityLabel = maxLabels.get(index);
		}else {
			majorityLabel = maxLabels.get(0);
		}
	}
		
	
	public Label predict(Instance instance) {		
			return majorityLabel;
	}
}