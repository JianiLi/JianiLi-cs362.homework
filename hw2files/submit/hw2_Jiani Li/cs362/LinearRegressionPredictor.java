package cs362;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections; 
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

public class LinearRegressionPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;

	private double w[][];
	private int featureNum;
	
	public void train(List<Instance> instances) throws NullPointerException{
		int instancesNum = instances.size();
		Instance instanceExample = instances.get(1);
		FeatureVector featureVectorExample = (FeatureVector)instanceExample.getFeatureVector();
		featureNum = featureVectorExample.featureVector.size()+1;

		double X[][] = new double[instancesNum][featureNum];
		
		double y[] = new double[instancesNum]; 
		int num = 0;
		for (Instance instance : instances){
			FeatureVector featureVector = (FeatureVector)instance.getFeatureVector();
			featureVector.add(0,1);
			List<Integer> keyList = new ArrayList<>(featureVector.featureVector.keySet());
			Collections.sort(keyList);

			for (int i = 0; i < keyList.size(); i++) {
				X[num][i] = featureVector.get(i);
			}	
			Label label = instance.getLabel();
			y[num] = Double.parseDouble(label.toString());
			num++;
		}

		RealMatrix XMatrix = new Array2DRowRealMatrix(X);
		//System.out.println("XMatrix.getColumnDimension()"+XMatrix.getColumnDimension()+"XMatrix.getRowDimension()"+XMatrix.getRowDimension());
		RealMatrix yMatrix = new Array2DRowRealMatrix(y);
		//System.out.println("yMatrix.getColumnDimension()"+yMatrix.getColumnDimension()+"yMatrix.getRowDimension()"+yMatrix.getRowDimension());
		w = new double[1][featureNum];
		w = (((inverseMatrix(XMatrix.transpose().multiply(XMatrix))).multiply(XMatrix.transpose())).multiply(yMatrix)).getData();
	}
		
	
	public Label predict(Instance instance) {
		
		FeatureVector featureVector = instance.getFeatureVector();
		featureVector.add(0,1);
		double x[] = new double[featureNum];
		for (int i = 0; i < x.length; i++) {
			x[i] = featureVector.get(i);
		}
		
		RealMatrix xMatrix = new Array2DRowRealMatrix(x);
		//System.out.println("xMatrix.getColumnDimension()"+xMatrix.getColumnDimension()+"xMatrix.getRowDimension()"+xMatrix.getRowDimension());
		RealMatrix wMatrix = new Array2DRowRealMatrix(w);
		//System.out.println("wMatrix.getColumnDimension()"+wMatrix.getColumnDimension()+"wMatrix.getRowDimension()"+wMatrix.getRowDimension());
		
		double prediction = ((wMatrix.transpose()).multiply(xMatrix)).getNorm();
		RegressionLabel linearRegressionLabel = new RegressionLabel(prediction);
		return linearRegressionLabel;	
	}
	 public static RealMatrix inverseMatrix(RealMatrix A) {
	        RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
	        return result; 
	    }
}
