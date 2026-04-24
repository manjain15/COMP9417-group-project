import kagglehub

# Download latest version
path = kagglehub.dataset_download("mytalkwithyou/bank-loan-approval-dataset")

print("Path to dataset files:", path)