import os 


print("Running Data Preprocessing...")
os.system("python data_preprocessing.py")

print("\n Training Model...")
os.system("python train_model.py")

print("\n Evaluating Model...")
os.system("python evaluate_model.py")

print("\n Complete Model Testing.")
