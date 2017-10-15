package cs362;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class Learn {

	static public LinkedList<Option> options = new LinkedList<Option>();
	public static double lambda = 1.0;
	public static double online_learning_rate = 1.0;
	public static int online_training_iterations = 1;
	public static double polynomial_kernel_exponent = 2;
	public static double gaussian_kernel_sigma = 1;
	public static double gradient_ascent_learning_rate = 0.01;
	public static int gradient_ascent_training_iterations = 5;
	public static double cluster_lambda = 0.0;
	public static int clustering_training_iterations = 10;
	
	
	public static void main(String[] args) throws IOException,NullPointerException,NumberFormatException {
			
		// Parse the command line.
		String[] manditory_args = { "mode"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Learn.options, manditory_args);
		// Naive Bayes smoothing parameter
		if (CommandLineUtilities.hasArg("lambda")){
			lambda = CommandLineUtilities.getOptionValueAsFloat("lambda");	
		}
		// Perceptron
		if (CommandLineUtilities.hasArg("online_learning_rate")) {
			online_learning_rate = CommandLineUtilities.getOptionValueAsFloat("online_learning_rate");
		}
		if (CommandLineUtilities.hasArg("online_training_iterations")) {
			online_training_iterations = CommandLineUtilities.getOptionValueAsInt("online_training_iterations");
		}
		String kernel = "linear_kernel";
		if (CommandLineUtilities.hasArg("kernel")) {
			kernel = CommandLineUtilities.getOptionValue("kernel");
			//System.out.println(kernel);
		}
		if (CommandLineUtilities.hasArg("polynomial_kernel_exponent")) {
			polynomial_kernel_exponent = CommandLineUtilities.getOptionValueAsFloat("polynomial_kernel_exponent");
		}
		if (CommandLineUtilities.hasArg("gaussian_kernel_sigma")) {
			gaussian_kernel_sigma = CommandLineUtilities.getOptionValueAsFloat("gaussian_kernel_sigma");
			//System.out.println(gaussian_kernel_sigma);
		}
		if (CommandLineUtilities.hasArg("gradient_ascent_learning_rate")) {
			gradient_ascent_learning_rate = CommandLineUtilities.getOptionValueAsFloat("gradient_ascent_learning_rate");
		}
		if (CommandLineUtilities.hasArg("gradient_ascent_training_iterations")) {
			//System.out.println(gradient_ascent_training_iterations);
			gradient_ascent_training_iterations = CommandLineUtilities.getOptionValueAsInt("gradient_ascent_training_iterations");
		}
		if (CommandLineUtilities.hasArg("cluster_lambda")) {
			cluster_lambda = CommandLineUtilities.getOptionValueAsFloat("cluster_lambda");
		}
		if (CommandLineUtilities.hasArg("clustering_training_iterations")) {
			clustering_training_iterations = CommandLineUtilities.getOptionValueAsInt("clustering_training_iterations");
		}
		
		String mode = CommandLineUtilities.getOptionValue("mode");
		String data = CommandLineUtilities.getOptionValue("data");
		String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
		String algorithm = CommandLineUtilities.getOptionValue("algorithm");
		String model_file = CommandLineUtilities.getOptionValue("model_file");
		String task = CommandLineUtilities.getOptionValue("task"); // classification vs. regression

		boolean classify = true;
		
		if (task != null && task.equals("regression")) {
		    classify = false;
		}
		
		if (mode.equalsIgnoreCase("train")) {
			if (data == null || algorithm == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, algorithm, model_file");
				System.exit(0);
			}
			// Load the training data.
			DataReader data_reader = new DataReader(data, classify);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Train the model.
			if (algorithm.equalsIgnoreCase("kernel_logistic_regression")) {
				//System.out.println(kernel);
				//System.out.println(polynomial_kernel_exponent);
				if(kernel.equalsIgnoreCase("linear_kernel")) {
					algorithm = "linear_kernel_logistic_regression";
				}else if(kernel.equalsIgnoreCase("polynomial_kernel")) {
					algorithm = "polynomial_kernel_logistic_regression";
				}else if(kernel.equalsIgnoreCase("gaussian_kernel")) {
					algorithm = "gaussian_kernel_logistic_regression";
				}
			}
			Predictor predictor = train(instances, algorithm, task);
			saveObject(predictor, model_file);		
			
		} else if (mode.equalsIgnoreCase("test")) {
			if (data == null || predictions_file == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, predictions_file, model_file");
				System.exit(0);
			}
			
			// Load the test data.
			DataReader data_reader = new DataReader(data, classify);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Load the model.
			Predictor predictor = (Predictor)loadObject(model_file);
			evaluateAndSavePredictions(predictor, instances, predictions_file, task);
		} else {
			System.out.println("Requires mode argument.");
		}
	}
	

	private static Predictor train(List<Instance> instances, String algorithm, String task) {
	    // TODO Train the model using "algorithm" on "data"
		Predictor predictor = null;
		
		if(algorithm.equalsIgnoreCase("majority")){
			predictor = new MajorityPredictor();
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("even_odd")){
			predictor = new EvenOddPredictor();
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("linear_regression")){
			predictor = new LinearRegressionPredictor();
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("naive_bayes")){
			predictor = new NaiveBayesClassifier(lambda);
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("perceptron")){
			predictor = new PerceptronPredictor(online_learning_rate, online_training_iterations);
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("linear_kernel_logistic_regression")){
			//System.out.println(algorithm);
			predictor = new LinearKernelLogisticRegression(gradient_ascent_learning_rate, gradient_ascent_training_iterations);
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("polynomial_kernel_logistic_regression")){
			//System.out.println(algorithm);
			predictor = new PolynomialKernelLogisticRegression(gradient_ascent_learning_rate, gradient_ascent_training_iterations, polynomial_kernel_exponent);
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("gaussian_kernel_logistic_regression")){
			//System.out.println(algorithm);
			predictor = new GaussianKernelLogisticRegression(gradient_ascent_learning_rate, gradient_ascent_training_iterations, gaussian_kernel_sigma);
			predictor.train(instances);
		}
		else if(algorithm.equalsIgnoreCase("lambda_means")){
			//System.out.println(algorithm);
			predictor = new LambdaMeansPredictor(cluster_lambda, clustering_training_iterations);
			predictor.train(instances);
		}
		else {
				System.out.println("Please enter a valid algorithm");
			}
		
		// TODO Evaluate the model
		Evaluator evaluator = new Evaluator();
		double accuracy = 0;
		if(task.equalsIgnoreCase("classification")){
			accuracy = evaluator.classsificationEvaluate(instances, predictor);
			if (!algorithm.equalsIgnoreCase("lambda_means")) {	
				System.out.println("The accuracy of the train set for "+ algorithm +" algorithm is "+accuracy);
			}
		}else if(task.equalsIgnoreCase("regression")){
			accuracy = evaluator.regressionEvaluate(instances, predictor);
			System.out.println("The mean error of the train set for "+ algorithm +" algorithm is "+accuracy);
		}
		
	    return predictor;
	}
		

	private static void evaluateAndSavePredictions(Predictor predictor,
			List<Instance> instances, String predictions_file, String task) throws IOException {
		PredictionsWriter writer = new PredictionsWriter(predictions_file);
		// TODO Evaluate the model if labels are available. 
		Evaluator evaluator = new Evaluator();
		double accuracy = 0;
		if(task.equalsIgnoreCase("classification")){
			//not output accuracy when lambda_means
			//accuracy = evaluator.classsificationEvaluate(instances, predictor);
			//System.out.println("The accuracy of the test set is "+ accuracy);
		}else if(task.equalsIgnoreCase("regression")){
			accuracy = evaluator.regressionEvaluate(instances, predictor);
			System.out.println("The mean error of the test set is "+ accuracy);
		}
				
		for (Instance instance : instances) {
			Label label = predictor.predict(instance);
			writer.writePrediction(label);
		}
		
		writer.close();	
	}

	public static void saveObject(Object object, String file_name) {
		try {
			ObjectOutputStream oos =
				new ObjectOutputStream(new BufferedOutputStream(
						new FileOutputStream(new File(file_name))));
			oos.writeObject(object);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + file_name + ": " + e);
		}
	}

	/**
	 * Load a single object from a filename. 
	 * @param file_name
	 * @return
	 */
	public static Object loadObject(String file_name) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(file_name))));
			Object object = ois.readObject();
			ois.close();
			return object;
		} catch (IOException e) {
			System.err.println("Error loading: " + file_name);
		} catch (ClassNotFoundException e) {
			System.err.println("Error loading: " + file_name);
		}
		return null;
	}
	
	public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		
		Learn.options.add(option);		
	}
	
	private static void createCommandLineOptions() {
		registerOption("data", "String", true, "The data to use.");
		registerOption("mode", "String", true, "Operating mode: train or test.");
		registerOption("predictions_file", "String", true, "The predictions file to create.");
		registerOption("algorithm", "String", true, "The name of the algorithm for training.");
		registerOption("model_file", "String", true, "The name of the model file to create/load.");
		// Other options will be added here.
		registerOption("task", "String", true, "The name of the task (classification or regression).");
		registerOption("lambda", "double", true, "The level of smoothing for Naive Bayes.");
		registerOption("online_learning_rate", "double", true, "The LTU learning rate.");
		registerOption("online_training_iterations", "int", true, "The number of training iterations for LTU.");
		registerOption("kernel", "String", true, "The kernel for kernel Logistic regression [linear_kernel, polynomial_kernel, gaussian_kernel].");
		registerOption("polynomial_kernel_exponent", "double", true, "The exponent of the polynomial kernel.");
		registerOption("gaussian_kernel_sigma", "double", true, "The sigma of the Gaussian kernel.");
		registerOption("gradient_ascent_learning_rate", "double", true, "The learning rate for logistic regression.");
		registerOption("gradient_ascent_training_iterations", "int", true, "The number of training iterations.");
		registerOption("cluster_lambda", "double", true, "The value of lambda in lambda-means.");
		registerOption("clustering_training_iterations", "int", true, "The number of lambda-means EM iterations.");
	}
}
