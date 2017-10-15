package cs362;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class LambdaMeansPredictor extends Predictor implements Serializable{
	
	private static final long serialVersionUID = 1L;
	private Double lambda;
	private Integer training_iterations;	
	private List<double[]> muList = new ArrayList<double[]>();	
	private Integer K;
	private Integer indexNum;
	

	public LambdaMeansPredictor(Double lambda, Integer training_iterations) {
		this.lambda = lambda;
		this.training_iterations = training_iterations;
	}
	
	public double getEuclideanDistSquare(double[] x1, double[] x2) {
		double dist = 0;		
		for (int i = 0; i < indexNum; i++) {			
			dist += Math.pow(x1[i] - x2[i], 2);
		}
		return dist;
	}
	
	public double[] getMeanVector(ArrayList<double[]> vectorList) {	
		int vectorNum = vectorList.size();
		double[] meanVector = new double[indexNum];
		for (int i = 0; i < indexNum; i++) {
			double mean = 0;
			for (int j = 0; j < vectorNum; j++) {
				mean += vectorList.get(j)[i];
			}
			mean = mean/vectorNum;
			meanVector[i] = mean;
		}
		return meanVector;
	}
		
	public void train(List<Instance> instances){
		
		//find the maximum feature index and feature vector list		
		indexNum = 0;
		ArrayList<FeatureVector> featureVectorList = new ArrayList<FeatureVector>();
		for (Instance instance : instances){
			FeatureVector featureVector = instance.getFeatureVector();
			featureVectorList.add(featureVector);
			int maxIndex = featureVector.getMaxIndex();
			if (maxIndex > indexNum) {
				indexNum = maxIndex;
			}
		}	
		int featureNum = featureVectorList.size();
		
		// transfer feature vector to array
		double[][] X = new double[featureNum][indexNum];
		for (int i = 0; i < featureNum; i++) {
			for (int j = 0; j < indexNum; j++) {
				X[i][j] = featureVectorList.get(i).get(j);
			}
		}
		
		// initialization, K = 1		
		K = 1;	
		double[] mu1 = new double[indexNum];
		ArrayList<double[]> X_list = new ArrayList<double[]>();
		for (int i = 0; i < featureNum; i++) {
			X_list.add(X[i]);
		}
		mu1 = getMeanVector(X_list);
		
		//get lambda value
		if (lambda == 0) {
			for (int i = 0; i < featureNum; i++) {
				lambda += getEuclideanDistSquare(X[i], mu1);
			}
			lambda /= featureNum;
		}
		//System.out.println(lambda);
				
		//inference with EM	
		muList.add(mu1);
		//System.out.println(muList);	
		
		for (int i = 0; i < training_iterations; i++) {
			int iter = i + 1;
			System.out.println("iteration:" + iter);
			//E-step
			int[] r = new int[featureNum];
			for(int fea = 0; fea < featureNum; fea++) {
				double minDist = Double.POSITIVE_INFINITY;
				int r_i = 0;
				for(int k = 0;k < K; k++) {
					double dist = getEuclideanDistSquare(X[fea], muList.get(k));
					if (dist < minDist) {
						minDist = dist;
						r_i = k+1;							
					}					
				}
				if (minDist <= lambda) {
					r[fea] = r_i;
				}else {
					K += 1;
					r[fea] = K;
					muList.add(X[fea]);
				}				
			}
			//System.out.println("E-step finishes");
			
			//M-step	
			muList.clear();
			for(int k = 1; k <= K; k++) {
				double[] mu_k;
				ArrayList<double[]> currentCluster = new ArrayList<double[]>();
				for (int fea = 0; fea < featureNum; fea++) {
					if (r[fea] == k) {
						currentCluster.add(X[fea]);
					}
				}	
				if (currentCluster.size()>0) {
					mu_k = getMeanVector(currentCluster);
				}else { //empty clusters
					mu_k = new double[indexNum];
				}
				muList.add(mu_k);
			}
			//System.out.println("M-step finishes");
		}	
		/*for (FeatureVector fea: featureVectorList) {
			System.out.println(fea.getCluster());
		}*/
		
	}
	
	public Label predict(Instance instance) {
		int cluster = 0;
		double minDist = Double.POSITIVE_INFINITY;
		FeatureVector featureVector = instance.getFeatureVector();
		double[] xi = new double[indexNum];
		for (int j = 0; j < indexNum; j++) {
			xi[j] = featureVector.get(j);
		}
		for(int k = 0;k < K; k++) {
			double dist = getEuclideanDistSquare(xi, muList.get(k));
			if (dist < minDist) {
				minDist = dist;
				cluster = k+1;							
			}					
		}
		ClassificationLabel clusterLabel = new ClassificationLabel(cluster);
		return clusterLabel;			
	}

}
