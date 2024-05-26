import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class PS4 {
	public static void main(String[] args) {
		
		double lamda = 0.001;
		double learningRate = 0.25; 
		String w1txt = args[0];
		String w2txt = args[1];
		String xdatatxt = args[2];
		String ydatatxt = args[3];

        
        
		double[][] w1 = readFile(w1txt, 30, 785);
		double[][] w2 = readFile(w2txt, 10, 31);
		double[][] xdata = readFile(xdatatxt, 10000, 784);
		double[][] ydata = readFile(ydatatxt, 10000, 1);
        double[][] hotYdata = new double[ydata.length][10];
        
        for (int i = 0; i < ydata.length; i++) {
            int value = (int) ydata[i][0];
            if (value == 0) {
                hotYdata[i][10 - 1] = 1;
            } else {
                hotYdata[i][value - 1] = 1;
            }
        }
        


		System.out.println("************************************************************");
		System.out.println("Problem Set:\t Problem Set 4: Neural Network");
		System.out.println("Name:\t\t Andrew Appleyard");
		System.out.println("Synax:\t\t java PS4 w1.txt w2.txt xdata.txt ydata.txt");
		System.out.println("************************************************************");

		System.out.println("\nTraining Phase: \t" + args[3]);
		System.out.println("--------------------------------------------------------------");
		System.out.println("\t=> Number of Entries (n):\t\t" + xdata.length);
		System.out.println("\t=> Number of Features (p):\t\t" + xdata[0].length);
		System.out.println("\n\nStarting Gradient Descent:");
		System.out.println("--------------------------------------------------------------");
		
		neutralNetwork(xdata, hotYdata, learningRate, w1, w2, ydata, lamda);
		
		saveWeights(w1, "w1out.txt");
        saveWeights(w2, "w2out.txt");
		
		System.out.println("\n\nTesting Phase (first 30 records):");
	    System.out.println("--------------------------------------------------------------");
	    testingPhase(xdata, hotYdata, w1, w2, ydata);

		
	}
	
	public static void neutralNetwork(double[][] X, double[][] Y, double alpha, double[][] W1, double[][] W2, double[][] YY, double lamda) {
	    int epoch = 0;
	    double prevLoss = Double.MAX_VALUE;
	    double accuracy = 0;
	    
	    try (BufferedWriter lossWriter = new BufferedWriter(new FileWriter("loss.txt")); 
	    	 BufferedWriter accuracyWriter = new BufferedWriter(new FileWriter("accuracy.txt"))) {
	    	
	    
	    while (epoch < 700) {
	        double[][] H1 = H1(X, W1);
	        double[][] Yhat = Yhat(H1, W2);

	        int correctPredictions = 0;
	        for (int i = 0; i < Yhat.length; i++) {
	        	
	            double maxProb = Double.MIN_VALUE;
	            int predictedClass = -1;
	            for (int j = 0; j < Yhat[0].length; j++) {
	            	
	                if (Yhat[i][j] > maxProb) {
	                    maxProb = Yhat[i][j];
	                    predictedClass = j+1;
	                }
	                
	            }
	            
	            if (predictedClass == (int) YY[i][0]) {
	                correctPredictions++;
	            }
	        }
	        accuracy = (double) correctPredictions / Y.length * 100.0;

	        double loss = calculateLoss(Y, Yhat, W1, W2, lamda);
	        double delta = (prevLoss - loss) / prevLoss * 100;
	        System.out.printf("Epoch %d:    Loss of %.2f    Delta = %.2f%%    Accuracy = %.2f%%\n", epoch + 1, loss, delta, accuracy);
	        prevLoss = loss;
	        
	        lossWriter.write(String.format("Epoch: %d \t Loss: %.3f\n", epoch + 1, loss));
            accuracyWriter.write(String.format("Epoch: %d \t Accuracy: %.3f\n", epoch + 1, accuracy));

	        double[][] delta2 = calculateDelta2(Yhat, Y);
	        double[][] delta1 = calculateDelta1(delta2, W2, X, W1);
	        double[][] gradient2 = calculateGradient2(delta2, H1, W2, lamda);
	        double[][] gradient1 = calculateGradient1(delta1, X, W1, lamda);

	        updateWeights(W1, W2, gradient1, gradient2, alpha);
	        epoch++;
	    }
	    System.out.println("Epochs Required: " + epoch);
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}
	
	
	
	public static void testingPhase(double[][] X, double[][] Y, double[][] W1, double[][] W2, double[][] YY) {
	    int correctPredictions = 0;
	    
	    for (int i = 0; i < 20; i++) {
	        double[][] H1 = H1(new double[][] { X[i] }, W1);
	        double[][] Yhat = Yhat(H1, W2);
	        int predictedClass = getPredictedClass(Yhat[0]);
	        int trueClass = (int) YY[i][0];
	        System.out.printf("Test Record %d: %d    Prediction: %d    Correct: %s\n", i + 1, trueClass, predictedClass, trueClass == predictedClass ? "TRUE" : "FALSE");
	        if (predictedClass == trueClass) {
	            correctPredictions++;
	        }
	    }
	    
	    System.out.println("\n=> Number of Test Entries (n): 20");
	    double accuracy = (double) correctPredictions / 20 * 100.0;
	    System.out.printf("=> Accuracy: %.0f%%\n", accuracy);
	}

	public static int getPredictedClass(double[] yhat) {
	    int predictedClass = -1;
	    double maxProb = Double.MIN_VALUE;
	    for (int i = 0; i < yhat.length; i++) {
	        if (yhat[i] > maxProb) {
	            maxProb = yhat[i];
	            predictedClass = i+1;
	        }
	    }
	    return predictedClass;
	}
	
	public static void saveWeights(double[][] weights, String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (double[] row : weights) {
                StringBuilder line = new StringBuilder();
                for (double weight : row) {
                    line.append(String.format("%.4f", weight)).append(",");
                }
                line.setLength(line.length() - 1);
                writer.write(line.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
	
	
	public static double calculateLoss(double[][] Y, double[][] Yhat, double[][] W1, double[][] W2, double lamda) {
        
        double loss = 0.0;
        
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < Y[0].length; j++) {
            	
            	
                double y = Y[i][j];
                double yhat = Yhat[i][j];
                
                loss += -y * Math.log(yhat) - (1 - y) * Math.log(1 - yhat);
                
            }
        }
        
        loss /= Y.length;

        double regTerm = 0.0;
        for (int j = 0; j < W1.length; j++) {
            for (int k = 1; k < W1[j].length; k++) {
                regTerm += Math.pow(W1[j][k], 2);
            }
        }
        for (int j = 0; j < W2.length; j++) {
            for (int k = 1; k < W2[j].length; k++) {
                regTerm += Math.pow(W2[j][k], 2);
            }
        }
        regTerm *= lamda / (2 * Y.length);

        
        loss += regTerm;

        return loss;
        
        
    }
	
	
	
	
	
	 public static void updateWeights(double[][] W1, double[][] W2, double[][] gradient1, double[][] gradient2, double alpha) {
		 
	        for (int i = 0; i < W2.length; i++) {
	            for (int j = 0; j < W2[i].length - 1; j++) {
	                W2[i][j] -= alpha * gradient2[i][j];
	            }
	        }

	        for (int i = 0; i < W1.length; i++) {
	            for (int j = 0; j < W1[i].length - 1; j++) {
	                W1[i][j] -= alpha * gradient1[i][j];
	            }
	        }
	 }
	 
	 
	 
	 
	
	public static double[][] calculateDelta2(double[][] Yhat, double[][] Y) {
        double[][] delta2 = new double[Y.length][Y[0].length];
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < Y[0].length; j++) {
            	
                delta2[i][j] = Yhat[i][j] - Y[i][j];
                
            }
        }
        return delta2;
    }
	
	
	
	
	

    public static double[][] calculateDelta1(double[][] delta2, double[][] W2, double[][] X, double[][] W1) {
        double[][] delta1 = new double[X.length][W1.length];
        
        double[][] delta2W2 = multiply(delta2, removeBias(W2));
        double[][] XW1 = matrixActivation(multiply(X, transpose(removeBias(W1))));
        
        
        
        for(int i = 0; i < X.length; i++) {
        	for(int j = 0; j < W1.length; j++) {
                delta1[i][j] =  delta2W2[i][j] * XW1[i][j];
        	}
        }
        
        return delta1;
    }
    
    
    
    
    
	
	public static double[][] Yhat(double[][] H1, double[][] W2) {
        
		double[][] biasH1 = addBias(H1, H1.length);
        
		//System.out.println("biasH1: ");
        //printDimensions(biasH1);
        //System.out.println("transposeW2: ");
        //printDimensions(transpose(W2));
        
        double[][] Yhat_temp = multiply(biasH1, transpose(W2));
        double[][] Yhat = new double[Yhat_temp.length][Yhat_temp[0].length];
        for (int i = 0; i < Yhat_temp.length; i++) {
            for (int j = 0; j < Yhat_temp[0].length; j++) {
                Yhat[i][j] = activationFunction(Yhat_temp[i][j]);
            }
        }

        return Yhat;
    }
	
	
	
	
	public static double[][] H1(double[][] X, double[][] W1) {
		
        double[][] biasX = addBias(X, X.length);
        
        //System.out.println("BiasX: ");
        //printDimensions(biasX);
        //System.out.println("transposeW1: ");
        //printDimensions(transpose(W1));
        
        double[][] H1_temp = multiply(biasX, transpose(W1));
        double[][] H1 = new double[H1_temp.length][H1_temp[0].length];
        for (int i = 0; i < H1_temp.length; i++) {
            for (int j = 0; j < H1_temp[0].length; j++) {
                H1[i][j] = activationFunction(H1_temp[i][j]);
            }
        }

        return H1;
    }
	
	
	
	
	public static double activationFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }
	
	
	
	
	public static double primeActivationFunction(double x) {
        return activationFunction(x) * (1 - activationFunction(x));
    }
	
	
	
	
	public static double[][] matrixActivation(double[][] x) {
        double[][] result = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = primeActivationFunction(x[i][j]);
            }
        }
        return result;
    }
	
	
	
	
	public static double[][] addBias(double[][] data, int rows) {
        int columns = data[0].length;
        double[][] newData = new double[rows][columns + 1];

        for (int i = 0; i < rows; i++) {
            newData[i][0] = 1;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                newData[i][j + 1] = data[i][j];
            }
        }

        return newData;
    }
	
	
	
	
	public static double[][] readFile(String filename, int rows, int columns){
        double[][] data = new double[rows][columns];

            String line;
            int row = 0;
            try {
            	
                BufferedReader br = new BufferedReader(new FileReader(filename));

				while ((line = br.readLine()) != null && row < rows) {
				    String[] values = line.split(",");

				    for (int col = 0; col < columns && col < values.length; col++) {
				        data[row][col] = Double.parseDouble(values[col]);
				    }
				    row++;
				}
			} catch (NumberFormatException | IOException e) {
				e.printStackTrace();
			}
        return data;
    }
	
	
	
	
	public static double[][] removeBias(double m[][]) {
		double[][] temp = new double[m.length][m[0].length - 1];

		for ( int i = 0; i < temp.length; i++ ) {
			for (int k = 0; k < temp[i].length; k++ ) {
				temp[i][k] = m[i][k+1];
			}
		}
		return temp;
	}
	
	
	

	public static double[][] transpose(double m[][]) {
		double[][] temp = new double[m[0].length][m.length];

		for (int i = 0; i < m[0].length; i++)
			for (int j = 0; j < m.length; j++)
				temp[i][j] = m[j][i];

		return temp;
	}
	
	
	
	
	public static double[][] calculateGradient2(double[][] delta2, double[][] H1, double[][] W2, double lamda) {
		
    	double[][] W2J = multiply(transpose(delta2), H1);
    	double[][] reg = new double[W2.length][W2[0].length];
    	double temp = 0;
    	for(int i = 0; i < W2.length; i++) {
    		for(int j = 0; j < W2[0].length; j++) {
    	    	temp += W2[i][j];
    		}
    	}
    	temp = temp * (lamda/10000);
    	
    	for(int i = 0; i < W2J.length; i++) {
    		for(int j = 0; j < W2J[0].length; j++) {
    			reg[i][j] = (W2J[i][j] + temp) / 10000;
    		}
    	}
    	
    	return reg;
        
    	
    	
    }
	
	
	

    public static double[][] calculateGradient1(double[][] delta1, double[][] X, double[][] W1, double lamda) {
    	
    	double[][] W1J = multiply(transpose(delta1), X);
    	double[][] reg = new double[W1.length][W1[0].length];
    	double temp = 0;
    	for(int i = 0; i < W1.length; i++) {
    		for(int j = 0; j < W1[0].length; j++) {
    	    	temp += W1[i][j];
    		}
    	}
    	temp = temp * (lamda/10000);
    	
    	for(int i = 0; i < W1J.length; i++) {
    		for(int j = 0; j < W1J[0].length; j++) {
    			reg[i][j] = (W1J[i][j] + temp) / 10000;
    		}
    	}
    	
    	return reg;
        
    }
    
    
    
    

	public static double[][] multiply(double[][] A, double[][] B) {

		int rows1 = A.length;
		int cols1 = A[0].length;
		int rows2 = B.length;
		int cols2 = B[0].length;

		if (cols1 != rows2) {
			System.err.println("Error with multiplication!  Check the dimensions.");
			throw new IllegalArgumentException();
		}

		double[][] C = new double[rows1][cols2];
		for (int i = 0; i < rows1; i++) {
			for (int j = 0; j < cols2; j++) {
				C[i][j] = 0.00000;
			}
		}

		for (int i = 0; i < rows1; i++) {
			for (int j = 0; j < cols2; j++) {
				for (int k = 0; k < cols1; k++) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}

		return C;
	}

	
	
	
	public static void printDimensions(double[][] m) {
		String xdim = String.format(" Matrix dimensions:    %d x %d ", m.length, m[0].length);

		System.out.println(xdim);
	}
}
