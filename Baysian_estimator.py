import numpy as np
import tkinter as tk
from Data_process import KNN
from tkinter import messagebox
from joblib import load



# Defining GUI class 

class ScanEntry:
    def __init__(self, master):
        self.master = master
        self.master.title("Scan Entry")
        
        self.label = tk.Label(master, text="Enter measurements in range (-100,-10) and comma-separated:")
        self.label.pack()
        
        self.entry = tk.Entry(master)  # Define the entry widget
        self.entry.pack()
        
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()
        
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def submit(self):
        new_scan_str = self.entry.get()
        measurements = new_scan_str.split(',')
        try:
            new_scan = [float(measurement.strip()) for measurement in measurements]
            location, probabilities = bayesian_estimator(new_scan)
            self.show_result(location, probabilities)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numbers separated by commas.")

    def show_result(self, location, probabilities):
        result_str = f"Location: {location+1}\nProbabilities:\n"
        for i, prob in enumerate(probabilities):
            result_str += f"loc{i+1}: {prob*100:.2f}%\n"
        self.result_label.config(text=result_str)

# Applying Baysian filter 

def bayesian_estimator(measurements):
    ## Set Prior probability distribution uniform 
        prior = np.ones(num_regions) / num_regions 
        bins = range(-100, -9)
        accuracy = 0
       

    ## Predict the labels for the test data
        predicted_location = knn.predict(measurements)

        prior[predicted_location] += 1
        prior /= np.sum(prior)

        while accuracy < 0.95:
            posterior = np.zeros(num_regions)

            for r in range(num_APs):
                index = bins.index(measurements[r])
                posterior += (prior * AP_likelihoods[index,r,:])

            posterior /= np.sum(posterior)  # Normalize

            max_prob = np.max(posterior)
            max_prob_indices = np.where(posterior == max_prob)[0]

            
            ## using the result of KNN if posterior has more than one maximum
            if len(max_prob_indices) == 1:
                max_prob_index = max_prob_indices[0]
            else:
            ## Use the prior probabilities to resolve ambiguity
                max_prob_index = np.argmax(prior[max_prob_indices])
                

            accuracy = max_prob

        # Update prior for next iteration
            prior = posterior


    
        return max_prob_index, posterior



# Definimg parameters
num_regions = 5
num_APs=3
APs = ["WiFi A", "WiFi B", "WiFi C"]
AP_likelihoods = np.load("pdf.npy", allow_pickle=True)
knn=load("knn_model.joblib")


def main():
    root = tk.Tk()
    app = ScanEntry(root)
    root.mainloop()

if __name__ == "__main__":
    main()




