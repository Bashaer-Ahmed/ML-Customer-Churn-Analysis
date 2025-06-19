"""
Customer Churn Data Analysis Module
BUS8405 Assignment - CLO1: Dataset Analysis and Feature Significance
Implementation: Comprehensive data exploration and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ChurnDataAnalyzer:
    """
    ML Data Analysis Implementation for Customer Churn Prediction
    Implements comprehensive data exploration and statistical analysis
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with dataset
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        self.numerical_features = []
        self.categorical_features = []
        
    def load_data(self):
        """Load and initial data inspection"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print("❌ Dataset not found. Please download from Kaggle first.")
            print("URL: https://www.kaggle.com/datasets/hassaneskikri/online-retail-customer-churn-dataset")
            return False
    
    def basic_info(self):
        """Generate basic dataset information"""
        if self.df is None:
            print("Please load data first")
            return
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"Dataset dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        print("\nColumn Information:")
        print("-" * 40)
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            print(f"{i:2d}. {col:<25} | {str(dtype):<10} | Nulls: {null_count}")
        
        # Identify feature types
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        if 'Customer_ID' in self.numerical_features:
            self.numerical_features.remove('Customer_ID')
        
        print(f"\nNumerical features ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        return {
            'shape': self.df.shape,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'null_counts': self.df.isnull().sum().to_dict()
        }
    
    def data_quality_assessment(self):
        """Assess data quality issues"""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Missing values
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        quality_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        })
        quality_df = quality_df[quality_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(quality_df) > 0:
            print("Missing Values Found:")
            print(quality_df)
        else:
            print("✅ No missing values found!")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Data types validation
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        return {
            'missing_values': quality_df.to_dict() if len(quality_df) > 0 else {},
            'duplicates': duplicates,
            'data_types': self.df.dtypes.to_dict()
        }
    
    def target_analysis(self):
        """Analyze target variable distribution"""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        target_col = 'Target_Churn'
        if target_col not in self.df.columns:
            print(f"Target column '{target_col}' not found!")
            return
        
        # Basic distribution
        churn_counts = self.df[target_col].value_counts()
        churn_percentages = self.df[target_col].value_counts(normalize=True) * 100
        
        print("Churn Distribution:")
        print("-" * 20)
        for value, count in churn_counts.items():
            percentage = churn_percentages[value]
            print(f"{value}: {count:4d} ({percentage:5.1f}%)")
        
        # Class balance assessment
        minority_class_ratio = min(churn_percentages) / 100
        if minority_class_ratio < 0.1:
            balance_status = "Severely Imbalanced"
        elif minority_class_ratio < 0.2:
            balance_status = "Moderately Imbalanced"
        elif minority_class_ratio < 0.4:
            balance_status = "Slightly Imbalanced"
        else:
            balance_status = "Balanced"
        
        print(f"\nClass Balance: {balance_status}")
        print(f"Minority class ratio: {minority_class_ratio:.3f}")
        
        return {
            'distribution': churn_counts.to_dict(),
            'percentages': churn_percentages.to_dict(),
            'balance_status': balance_status,
            'minority_ratio': minority_class_ratio
        }
    
    def feature_statistics(self):
        """Comprehensive statistical analysis of features"""
        print("\n" + "="*60)
        print("FEATURE STATISTICAL ANALYSIS")
        print("="*60)
        
        # Numerical features statistics
        if self.numerical_features:
            print("\nNumerical Features Summary:")
            print("-" * 30)
            numerical_stats = self.df[self.numerical_features].describe()
            print(numerical_stats.round(2))
            
            # Additional statistics
            print("\nAdditional Statistics:")
            for feature in self.numerical_features:
                skewness = self.df[feature].skew()
                kurtosis = self.df[feature].kurtosis()
                print(f"{feature:25s} | Skewness: {skewness:6.2f} | Kurtosis: {kurtosis:6.2f}")
        
        # Categorical features analysis
        if self.categorical_features:
            print(f"\nCategorical Features Analysis:")
            print("-" * 30)
            for feature in self.categorical_features:
                if feature != 'Customer_ID':
                    unique_count = self.df[feature].nunique()
                    most_common = self.df[feature].mode()[0]
                    most_common_count = self.df[feature].value_counts().iloc[0]
                    print(f"{feature:25s} | Unique: {unique_count:3d} | Mode: {most_common} ({most_common_count})")
        
        return {
            'numerical_stats': numerical_stats.to_dict() if self.numerical_features else {},
            'categorical_summary': {feature: {
                'unique_count': self.df[feature].nunique(),
                'mode': self.df[feature].mode()[0] if len(self.df[feature].mode()) > 0 else None
            } for feature in self.categorical_features if feature != 'Customer_ID'}
        }
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        if len(self.numerical_features) < 2:
            print("Not enough numerical features for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = self.df[self.numerical_features].corr()
        
        print("Correlation Matrix (Top correlations with Target_Churn):")
        print("-" * 50)
        
        if 'Target_Churn' in correlation_matrix.columns:
            target_correlations = correlation_matrix['Target_Churn'].abs().sort_values(ascending=False)
            target_correlations = target_correlations[target_correlations.index != 'Target_Churn']
            
            print("Features most correlated with churn:")
            for feature, corr in target_correlations.head(10).items():
                direction = "positive" if correlation_matrix.loc[feature, 'Target_Churn'] > 0 else "negative"
                print(f"{feature:25s} | {corr:6.3f} ({direction})")
        
        # Find high correlations between features (multicollinearity)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"\nHigh correlations between features (>0.7):")
            for pair in high_corr_pairs:
                print(f"{pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("\n✅ No high correlations found between features")
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'target_correlations': target_correlations.to_dict() if 'Target_Churn' in correlation_matrix.columns else {},
            'high_correlations': high_corr_pairs
        }
    
    def feature_importance_analysis(self):
        """Analyze feature importance for business insights"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE FOR BUSINESS INSIGHTS")
        print("="*60)
        
        insights = {}
        
        # Customer Demographics Analysis
        print("\n1. DEMOGRAPHIC INSIGHTS:")
        print("-" * 25)
        
        if 'Age' in self.df.columns:
            age_churn = self.df.groupby('Target_Churn')['Age'].agg(['mean', 'median', 'std'])
            print("Age analysis:")
            print(age_churn.round(2))
            insights['age_analysis'] = age_churn.to_dict()
        
        if 'Gender' in self.df.columns:
            gender_churn = pd.crosstab(self.df['Gender'], self.df['Target_Churn'], normalize='columns') * 100
            print(f"\nGender distribution by churn:")
            print(gender_churn.round(1))
            insights['gender_analysis'] = gender_churn.to_dict()
        
        # Financial Behavior Analysis
        print("\n2. FINANCIAL BEHAVIOR INSIGHTS:")
        print("-" * 30)
        
        financial_features = ['Annual_Income', 'Total_Spend', 'Average_Transaction_Amount']
        for feature in financial_features:
            if feature in self.df.columns:
                finance_stats = self.df.groupby('Target_Churn')[feature].agg(['mean', 'median'])
                print(f"\n{feature}:")
                print(finance_stats.round(2))
                insights[f'{feature.lower()}_analysis'] = finance_stats.to_dict()
        
        # Customer Relationship Analysis
        print("\n3. CUSTOMER RELATIONSHIP INSIGHTS:")
        print("-" * 32)
        
        relationship_features = ['Years_as_Customer', 'Satisfaction_Score', 'Num_of_Support_Contacts']
        for feature in relationship_features:
            if feature in self.df.columns:
                rel_stats = self.df.groupby('Target_Churn')[feature].agg(['mean', 'median'])
                print(f"\n{feature}:")
                print(rel_stats.round(2))
                insights[f'{feature.lower()}_analysis'] = rel_stats.to_dict()
        
        # Engagement Analysis
        print("\n4. ENGAGEMENT INSIGHTS:")
        print("-" * 20)
        
        if 'Email_Opt_In' in self.df.columns:
            email_churn = pd.crosstab(self.df['Email_Opt_In'], self.df['Target_Churn'], normalize='columns') * 100
            print("Email opt-in by churn:")
            print(email_churn.round(1))
            insights['email_analysis'] = email_churn.to_dict()
        
        if 'Promotion_Response' in self.df.columns:
            promo_churn = pd.crosstab(self.df['Promotion_Response'], self.df['Target_Churn'], normalize='columns') * 100
            print(f"\nPromotion response by churn:")
            print(promo_churn.round(1))
            insights['promotion_analysis'] = promo_churn.to_dict()
        
        return insights
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn distribution
        churn_counts = self.df['Target_Churn'].value_counts()
        ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Customer Churn Distribution')
        
        # Age distribution by churn
        if 'Age' in self.df.columns:
            self.df.boxplot(column='Age', by='Target_Churn', ax=ax2)
            ax2.set_title('Age Distribution by Churn Status')
            ax2.set_xlabel('Churn Status')
        
        # Satisfaction score by churn
        if 'Satisfaction_Score' in self.df.columns:
            satisfaction_churn = self.df.groupby(['Satisfaction_Score', 'Target_Churn']).size().unstack(fill_value=0)
            satisfaction_churn.plot(kind='bar', ax=ax3)
            ax3.set_title('Satisfaction Score vs Churn')
            ax3.set_xlabel('Satisfaction Score')
            ax3.legend(['No Churn', 'Churn'])
        
        # Spending behavior
        if 'Total_Spend' in self.df.columns:
            self.df.boxplot(column='Total_Spend', by='Target_Churn', ax=ax4)
            ax4.set_title('Total Spend by Churn Status')
            ax4.set_xlabel('Churn Status')
        
        plt.tight_layout()
        plt.savefig('results/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation heatmap
        if len(self.numerical_features) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[self.numerical_features].corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Visualizations saved to results/ folder")
    
    def run_complete_analysis(self):
        """Run the complete data analysis pipeline"""
        print("🚀 Starting Comprehensive Data Analysis")
        print("="*70)
        
        # Load data
        if not self.load_data():
            return None
        
        # Run all analyses
        results = {}
        results['basic_info'] = self.basic_info()
        results['data_quality'] = self.data_quality_assessment()
        results['target_analysis'] = self.target_analysis()
        results['feature_statistics'] = self.feature_statistics()
        results['correlation_analysis'] = self.correlation_analysis()
        results['feature_importance'] = self.feature_importance_analysis()
        
        # Generate visualizations
        self.generate_visualizations()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nKey Findings Summary:")
        print(f"• Dataset size: {results['basic_info']['shape']}")
        print(f"• Target distribution: {results['target_analysis']['balance_status']}")
        print(f"• Data quality: {'Good' if results['data_quality']['duplicates'] == 0 else 'Issues found'}")
        print("• Visualizations saved to results/ folder")
        
        return results

def main():
    """Main function to run data analysis"""
    # Initialize analyzer
    analyzer = ChurnDataAnalyzer('data/raw/online_retail_customer_churn.csv')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n🎯 Analysis results available for assignment questions!")
        return analyzer, results
    else:
        print("\n❌ Please download the dataset first and place it in data/raw/")
        return None, None

if __name__ == "__main__":
    analyzer, results = main()
