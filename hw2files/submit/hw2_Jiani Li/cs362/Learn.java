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

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Learn {

	static public LinkedList<Option> options = new LinkedList<Option>();
	
	public static void main(String[] args) throws IOException,NullPointerException,NumberFormatException {
				
		// Parse the command line.
		String[] manditory_args = { "mode"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Learn.options, manditory_args);
		// Naive Bayes smoothing parameter
		double lambda = 1.0;
		if (CommandLineUtilities.hasArg("lambda")){
			lambda = CommandLineUtilities.getOptionValueAsFloat("lambda");	
		}
		// Perceptron
		double online_learning_rate = 1.0;
		if (CommandLineUtilities.hasArg("online_learning_rate")) {
			online_learning_rate = CommandLineUtilities.getOptionValueAsFloat("online_learning_rate");
		}
		int online_training_iterations = 1;
		if (CommandLineUtilities.hasArg("online_training_iterations")) {
			online_training_iterations = CommandLineUtilities.getOptionValueAsInt("online_training_iterations");
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
			Predictor predictor = train(instances, algorithm, lambda, online_learning_rate, online_training_iterations, task);
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
	

	private static Predictor train(List<Instance> instances, String algorithm, Double lambda, Double online_learning_rate, Integer online_training_iterations, String task) {
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
		else {
				System.out.println("Please enter a valid algorithm");
			}
		
		// TODO Evaluate the model
		Evaluator evaluator = new Evaluator();
		double accuracy = 0;
		if(task.equalsIgnoreCase("classification")){
			accuracy = evaluator.classsificationEvaluate(instances, predictor);
			System.out.println("The accuracy of the train set for "+ algorithm +" algorithm is "+accuracy);
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
			accuracy = evaluator.classsificationEvaluate(instances, predictor);
			System.out.println("The accuracy of the test set is "+ accuracy);
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
		registerOption("task", "String", true, "The name of the task (classification or regression).");
		registerOption("lambda", "double", true, "The level of smoothing for Naive Bayes.");
		registerOption("online_learning_rate", "double", true, "The LTU learning rate.");
		registerOption("online_training_iterations", "int", true, "The number of training iterations for LTU.");
		// Other options will be added here.
	}
}
