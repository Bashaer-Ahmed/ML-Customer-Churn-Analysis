"""
Dataset Download Script for Customer Churn ML Project
BUS8405 Assignment - Data Acquisition Helper
"""

import os
import urllib.request
import pandas as pd

def download_dataset():
    """Download the customer churn dataset for analysis"""
    
    print("📥 Downloading Customer Churn Dataset for ML Analysis...")
    print("="*60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Dataset URL (if available as direct download)
    # Note: Kaggle datasets usually require API authentication
    dataset_url = "https://www.kaggle.com/datasets/hassaneskikri/online-retail-customer-churn-dataset"
    
    print("🔗 Dataset Source:")
    print(f"   {dataset_url}")
    
    print("\n📋 Manual Download Instructions:")
    print("1. Visit the Kaggle link above")
    print("2. Click 'Download' to get the CSV file")
    print("3. Save as: data/raw/online_retail_customer_churn.csv")
    
    print("\n🔧 Alternative: Using Kaggle API")
    print("1. Install kaggle: pip install kaggle")
    print("2. Setup API credentials from Kaggle account settings")
    print("3. Run: kaggle datasets download -d hassaneskikri/online-retail-customer-churn-dataset")
    print("4. Extract to data/raw/ folder")
    
    # Check if file already exists
    if os.path.exists('data/raw/online_retail_customer_churn.csv'):
        print("\n✅ Dataset already exists!")
        
        # Verify the dataset
        try:
            df = pd.read_csv('data/raw/online_retail_customer_churn.csv')
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print("\n⚠️  Dataset not found. Please download manually.")
        return False

def create_sample_dataset():
    """Create a sample dataset for testing (if original not available)"""
    print("\n🔧 Creating sample dataset for testing...")
    
    import numpy as np
    np.random.seed(42)
    
    # Generate synthetic data similar to the original
    n_samples = 1000
    
    data = {
        'Customer_ID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Annual_Income': np.random.uniform(20, 200, n_samples),
        'Total_Spend': np.random.uniform(100, 10000, n_samples),
        'Years_as_Customer': np.random.randint(1, 20, n_samples),
        'Num_of_Purchases': np.random.randint(1, 100, n_samples),
        'Average_Transaction_Amount': np.random.uniform(10, 500, n_samples),
        'Num_of_Returns': np.random.randint(0, 10, n_samples),
        'Num_of_Support_Contacts': np.random.randint(0, 5, n_samples),
        'Satisfaction_Score': np.random.randint(1, 6, n_samples),
        'Last_Purchase_Days_Ago': np.random.randint(1, 365, n_samples),
        'Email_Opt_In': np.random.choice([True, False], n_samples),
        'Promotion_Response': np.random.choice(['Responded', 'Ignored', 'Unsubscribed'], n_samples),
        'Target_Churn': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Lower satisfaction should increase churn probability
    high_churn_mask = df['Satisfaction_Score'] <= 2
    df.loc[high_churn_mask, 'Target_Churn'] = np.random.choice([True, False], 
                                                              sum(high_churn_mask), 
                                                              p=[0.7, 0.3])
    
    # Save sample dataset
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/online_retail_customer_churn.csv', index=False)
    
    print(f"✅ Sample dataset created: {df.shape}")
    print("   Note: This is synthetic data for testing purposes")
    print("   For the actual assignment, please use the real Kaggle dataset")
    
    return True

def main():
    """Main function"""
    print("🚀 Customer Churn Dataset Setup")
    print("="*40)
    
    # Try to download/verify the real dataset
    if not download_dataset():
        response = input("\nWould you like to create a sample dataset for testing? (y/n): ")
        if response.lower() == 'y':
            create_sample_dataset()
        else:
            print("Please download the dataset manually before running the analysis.")

if __name__ == "__main__":
    main()
