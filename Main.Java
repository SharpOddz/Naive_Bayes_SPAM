import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;

public class Main {

	//Raw Full Data, 4601 training instances and 57 features
	double[][] all_data = new double[4601][57];
	int[][] all_data_label = new int[4601][1];
	
	//Training set data (2301 training instances)
	double[][] training_data = new double[2301][57];
	int[][] training_data_label = new int[2301][1];
	
	//Test set data (2300 training instances)
	double[][] test_data = new double[2300][57];
	int[][] test_data_label = new int[2300][1];
	
	//Storing mean and STDEV of each feature within the training set, seperated by class
	double[][] training_data_spam_mean = new double[57][1];
	double[][] training_data_spam_stdev = new double[57][1];
	double[][] training_data_not_spam_mean = new double[57][1];
	double[][] training_data_not_spam_stdev = new double[57][1];
	
	//Instances of Spam and Not Spam in the training data
	int training_spam_count = 0;
	int training_not_spam_count = 0;
	
	public static void main(String[] args) throws Exception{
		Main m = new Main();
	}
	
	public Main() throws Exception{
		//Getting all the data
		readData();
		//Generate the training and test set data (2301 and 2300 training instances respective)
		generateTrainingTestSets();
		//[Part 2 of HW]
			//Calculate Prior Probability of training data 
			calculate_prior_probability_training_data();
			//Calculate Mean and STDEV of each feature
			calculate_mean_stdev_training_data();
		//[Part 3 of HW]
			//Calculate Gaussian probability function
			//Classify based on gaussian probability
			calculate_test_set_accuracy();
	}
	
	private void calculate_test_set_accuracy() {
		//Go through each test data instance
			//run it through the classify function (give this function the test data instance array and log of p(spam) and p(not spam)
		//The probability of each class occuring in the training data was found in the function 
			//Calculate Prior probability training data and stored as global variables 
		double training_spam_log = Math.log((double)training_spam_count / (double)training_data.length);
		double training_not_spam_log = Math.log((double)training_not_spam_count / (double)training_data.length);
		
		//Tracking Confusion Matrix as the program makes predictions
		int true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
		
		
		for(int i = 0;i < test_data.length;i++) {
			int prediction = classify(test_data[i],training_spam_log,training_not_spam_log);
			//If the prediction is the same as the label then it correctly classified it
				//But I want to know what is true positive, false positive...
			if(test_data_label[i][0] == 1 && prediction == 1) {
				true_positive++;
			}
			if(test_data_label[i][0] == 1 && prediction == 0) {
				false_negative++;
			}
			if(test_data_label[i][0] == 0 && prediction == 1) {
				false_positive++;
			}
			if(test_data_label[i][0] == 0 && prediction == 0) {
				true_negative++;
			}
		}
		
		//Confusion Matrix
		double accuracy = ((double)true_positive + (double)true_negative) / ((double)test_data.length);
		double recall = ((double)true_positive) / ((double)true_positive + (double)false_negative);
		double precision = ((double)true_positive) / ((double)true_positive + (double)false_positive);
		
		System.out.println("Accuracy: " + accuracy);
		System.out.println("Recall: " + recall);
		System.out.println("Precision: " + precision);
		
		System.out.println("TP: " + true_positive + " TN: " + true_negative + " FP: " + false_positive + " FN: " + false_negative);
		
	}
	
	private int classify(double[] features, double log_spam, double log_not_spam) {
		//Go through each feature, add the gaussian probability of each feature for both spam 
			//and not spam classes 
		for(int i = 0;i < features.length;i++) {
			log_spam += calculate_gaussian_probability(features[i],training_data_spam_mean[i][0],training_data_spam_stdev[i][0]);
			log_not_spam += calculate_gaussian_probability(features[i],training_data_not_spam_mean[i][0],training_data_not_spam_stdev[i][0]);
		}
		//Classify as the class with the highest probability
		if(log_spam > log_not_spam) {
			return 1;
		}
		else {
			return 0;
		}
	}
	
	private double calculate_gaussian_probability(double x, double u, double stdev) {
		//u is the mean 
		//Since we are taking the log of this it will simplify to -ln(SQRT(2pi)*stdev) + -((x-u)^2 / 2*STDEV^2)
		//Numerator: e^-((x-u)^2 / 2*STDEV^2)
			//But since we are taking the log we can get rid of the e
			double numerator_exponent = -(Math.pow(x-u,2) / (2*Math.pow(stdev, 2)));
		//Denominator
			//Since we are taking the log of this, it is simply the log of SQRT(2pi)*Stdev but times -1
			double denominator = (Math.sqrt(2*Math.PI)*stdev);
			denominator = -Math.log(denominator);
		//System.out.println(denominator + numerator_exponent);
		return denominator + numerator_exponent;
	}

	
	private void calculate_mean_stdev_training_data() {
		int spam = 0, not_spam = 0;
		//Method for obtaining mean:
			//I will first sum each feature up and then divide by number of training instances given the class
		for(int i = 0;i < training_data.length;i++) {
			//Counting instances of each class
			if(training_data_label[i][0] == 0) {
				for(int a = 0;a < training_data[i].length;a++) {
					training_data_not_spam_mean[a][0] += training_data[i][a];
				}
				not_spam++;
			}
			else {
				for(int a = 0;a < training_data[i].length;a++) {
					training_data_spam_mean[a][0] += training_data[i][a];
				}
				spam++;
			}
		}
		//To get the mean, I will divide the sum by the number of training instances
			//Seperating by class as well
		for(int i = 0;i < training_data_spam_mean.length;i++) {
			training_data_spam_mean[i][0] = (training_data_spam_mean[i][0] / (double)spam); 
			training_data_not_spam_mean[i][0] = (training_data_not_spam_mean[i][0] / (double)not_spam); 
		}
		
		//Method for obtaining STDEV:
			//Now that I have the mean of each variable (class-specific)
			//I can calculate variance through the sum of squared differences and then dividing by 
			//Once I have variance I take the square root to obtain STDEV
		double sum_squared_difference = 0;
		for(int i = 0;i < training_data.length;i++) {
			//Not Spam
			if(training_data_label[i][0] == 0) {
				for(int a = 0;a < training_data[i].length;a++) {
					training_data_not_spam_stdev[a][0] += Math.pow((training_data[i][a] - training_data_not_spam_mean[a][0]), 2);
				}
			}
			//Spam
			else {
				for(int a = 0;a < training_data[i].length;a++) {
					training_data_spam_stdev[a][0] += Math.pow((training_data[i][a] - training_data_spam_mean[a][0]), 2);
				}
			}
		}
		
			//Last 2 steps is to divide the sum of squared difference by number of instances
			//And then take the square root
			//I will check for zero STDEV values after these lines of code	
		for(int i = 0; i < training_data_spam_stdev.length;i++) {
			training_data_spam_stdev[i][0] = ((training_data_spam_stdev[i][0]) / (double)spam);
			training_data_spam_stdev[i][0] = Math.sqrt(training_data_spam_stdev[i][0]);
			training_data_not_spam_stdev[i][0] = ((training_data_not_spam_stdev[i][0]) / (double)not_spam);
			training_data_not_spam_stdev[i][0] = Math.sqrt(training_data_not_spam_stdev[i][0]);
		}
		
		//Also check to see if any STDEV values are 0, in which case I will assign it a small (0.001) value
		for(int i = 0; i < training_data_spam_stdev.length;i++) {
			if(training_data_spam_stdev[i][0] == 0) {
				training_data_spam_stdev[i][0] = 0.001;
			}
			if(training_data_not_spam_stdev[i][0] == 0) {
				training_data_not_spam_stdev[i][0] = 0.001;
			}
		}
		
		//Printout
		/*
		for(int i = 0; i < training_data_spam_mean.length;i++) {
			System.out.println("(SPAM) Feature " + i + " , Mean: " + training_data_spam_mean[i][0] + " , STDEV: " + training_data_spam_stdev[i][0]);
			System.out.println("(NOT SPAM) Feature " + i + " , Mean: " + training_data_not_spam_mean[i][0] + " , STDEV: " + training_data_not_spam_stdev[i][0]);
		}
		*/
		
	}
	
	//Calculating the prior probability of the two classes in the training data
	private void calculate_prior_probability_training_data() {
		//First get a count of each class
		int spam = 0, not_spam = 0;
		for(int i = 0;i < training_data.length;i++) {
			if(training_data_label[i][0] == 0) {
				not_spam++;
			}
			else {
				spam++;
			}
		}
		//Once count is gotten then I can divide by the total number of instances within the training set
		double spam_probability = ((double)spam)/training_data.length;
		double not_spam_probability = ((double)not_spam)/training_data.length;
		
		training_spam_count = spam;
		training_not_spam_count = not_spam;
		
		System.out.println("Spam Probability In Training Dataset: " + spam_probability);
		System.out.println("Not Spam Probability In Training Dataset: " + not_spam_probability);
	}
	
	private void generateTrainingTestSets() {
		//First find the indices that are spam and those that are not
		ArrayList<Integer> spam_indices = new ArrayList<>();
		ArrayList<Integer> not_spam_indices = new ArrayList<>();
		for(int i = 0;i < all_data_label.length;i++) {
			if(all_data_label[i][0] == 0) {
				not_spam_indices.add(i);
			}
			else {
				spam_indices.add(i);
			}
		}
		
		//I can use the Collections.shuffle method to randomize the arraylists
		Collections.shuffle(spam_indices);
		Collections.shuffle(not_spam_indices);
		
		//Fill Training and Test Sets, (First with spam and then with non-spam)
			//Spam: (2301 * 0.4) = 920
			//Non-Spam: (2301 * 0.6) = 1381
	    int training_spam_count = (int) (2301 * 0.4); 
	    int training_not_spam_count = 2301 - training_spam_count; 
	    
	    int test_spam_count = (int) (2300 * 0.4); 
	    int test_not_spam_count = 2300 - test_spam_count; 
	    
	    //First the training set
	    int index = 0;
		for(int i = 0;i < training_spam_count;i++) {
			//Copy appropiate array
			System.arraycopy(all_data[spam_indices.get(i)], 0, training_data[i], 0, 57);
			training_data_label[i][0] = all_data_label[spam_indices.get(i)][0];
			index++;
		}
		for(int i = 0;i < training_not_spam_count;i++) {
			System.arraycopy(all_data[not_spam_indices.get(i)], 0, training_data[index], 0, 57);
			training_data_label[index][0] = all_data_label[not_spam_indices.get(i)][0];
			index++;
		}
		
        //Test Set, reset index to zero as the test set tracker
		index = 0;
		for(int i = training_spam_count;i < spam_indices.size();i++) {
			//Copy appropiate array
			System.arraycopy(all_data[spam_indices.get(i)], 0, test_data[index], 0, 57);
			test_data_label[index][0] = all_data_label[spam_indices.get(i)][0];
			index++;
		}
		for(int i = training_not_spam_count;i < not_spam_indices.size();i++) {
			//Copy appropiate array
			System.arraycopy(all_data[not_spam_indices.get(i)], 0, test_data[index], 0, 57);
			test_data_label[index][0] = all_data_label[not_spam_indices.get(i)][0];
			index++;
		}
		
		//Verify the training and test set have the correct distribution
		/*
		int spam = 0, not_spam = 0;
		for(int i = 0;i < training_data.length;i++) {
			if(training_data_label[i][0] == 0 ) {
				not_spam++;
			}
			else {
				spam++;
			}
		}
		System.out.println("Test Set,  Spam: " + spam + "(" + (((double)spam/2300)*100) +  "%) , Not Spam: " + not_spam + "(" + (((double)not_spam/2300)*100) +  "%)");
		spam = 0;
		not_spam = 0;
		for(int i = 0;i < test_data.length;i++) {
			if(test_data_label[i][0] == 0 ) {
				not_spam++;
			}
			else {
				spam++;
			}
		}
		System.out.println("Test Set,  Spam: " + spam + "(" + (((double)spam/2300)*100) +  "%) , Not Spam: " + not_spam + "(" + (((double)not_spam/2300)*100) +  "%)");
		*/
	}
	
	private void readData() throws Exception{
		//Reading the raw data
		BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\sharp\\OneDrive\\Desktop\\spambase\\spambase.data"));
		String line;
        int index = 0;
        while ((line = br.readLine()) != null) {
            String[] str_arr = line.split(",");
            for(int i = 0;i < str_arr.length-1;i++) {
            	all_data[index][i] = Double.parseDouble(str_arr[i]);
            }
            all_data_label[index][0] = Integer.parseInt(str_arr[57]);
            //System.out.println(line);
        	index++;
        }
        System.out.println(index);
	}

}
