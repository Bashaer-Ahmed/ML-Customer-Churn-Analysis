"""
ML Model Comparison and Selection Module
BUS8405 Assignment - CLO2: Machine Learning Method Selection and Evaluation
Implementation: Comprehensive comparison of 9 ML algorithms with justified selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MLModelComparator:
    """
    Comprehensive ML model comparison for churn prediction
    """
    
    def __init__(self, data_path):
        """
        Initialize model comparator
        
        Args:
            data_path (str): Path to the dataset
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            
            # Basic data preparation
            self._prepare_features()
            self._split_data()
            
            return True
        except FileNotFoundError:
            print("❌ Dataset not found. Please ensure the dataset is in data/raw/")
            return False
    
    def _prepare_features(self):
        """Prepare features for modeling"""
        print("\n" + "="*50)
        print("FEATURE PREPARATION")
        print("="*50)
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Remove customer ID if present
        if 'Customer_ID' in df_processed.columns:
            df_processed = df_processed.drop('Customer_ID', axis=1)
        
        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Label encode categorical variables
        label_encoders = {}
        for col in categorical_columns:
            if col != 'Target_Churn':  # Don't encode target
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        # Convert Target_Churn to numeric if it's boolean/string
        if 'Target_Churn' in df_processed.columns:
            if df_processed['Target_Churn'].dtype == 'object' or df_processed['Target_Churn'].dtype == 'bool':
                df_processed['Target_Churn'] = df_processed['Target_Churn'].astype(str)
                df_processed['Target_Churn'] = (df_processed['Target_Churn'] == 'True').astype(int)
        
        # Separate features and target
        self.y = df_processed['Target_Churn']
        self.X = df_processed.drop('Target_Churn', axis=1)
        
        print(f"\nFeatures shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Feature columns: {list(self.X.columns)}")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
        
        return label_encoders
    
    def _split_data(self):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale numerical features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData split completed:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        print(f"Training target distribution: {self.y_train.value_counts().to_dict()}")
        print(f"Testing target distribution: {self.y_test.value_counts().to_dict()}")
    
    def initialize_models(self):
        """Initialize various ML models for comparison"""
        print("\n" + "="*50)
        print("INITIALIZING ML MODELS")
        print("="*50)
        
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'scaled': True,
                'description': 'Linear model for binary classification with probabilistic output'
            },
            
            'Random Forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'scaled': False,
                'description': 'Ensemble of decision trees with voting mechanism'
            },
            
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'scaled': False,
                'description': 'Sequential ensemble learning with error correction'
            },
            
            'Support Vector Machine': {
                'model': SVC(kernel='rbf', probability=True, random_state=42),
                'scaled': True,
                'description': 'Maximum margin classifier with kernel trick'
            },
            
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(n_neighbors=5),
                'scaled': True,
                'description': 'Instance-based learning using distance metrics'
            },
            
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42, max_depth=10),
                'scaled': False,
                'description': 'Tree-based model with interpretable rules'
            },
            
            'Naive Bayes': {
                'model': GaussianNB(),
                'scaled': True,
                'description': 'Probabilistic classifier based on Bayes theorem'
            },
            
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'scaled': False,
                'description': 'Optimized gradient boosting framework'
            },
            
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'scaled': False,
                'description': 'Fast gradient boosting with leaf-wise tree growth'
            }
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name, info in self.models.items():
            print(f"• {name}: {info['description']}")
    
    def evaluate_model(self, name, model_info, cv_folds=5):
        """Evaluate a single model with cross-validation"""
        model = model_info['model']
        use_scaled = model_info['scaled']
        
        # Choose appropriate data
        X_train = self.X_train_scaled if use_scaled else self.X_train
        X_test = self.X_test_scaled if use_scaled else self.X_test
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, self.y_train, 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='accuracy', n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            return metrics, y_pred, y_pred_proba
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            return None, None, None
    
    def compare_all_models(self):
        """Compare all initialized models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        self.results = {}
        
        for name, model_info in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            metrics, y_pred, y_pred_proba = self.evaluate_model(name, model_info)
            
            if metrics is not None:
                self.results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'model': model_info['model'],
                    'description': model_info['description']
                }
                
                print(f"  ✅ Accuracy: {metrics['accuracy']:.4f}")
                print(f"     F1-Score: {metrics['f1_score']:.4f}")
                print(f"     ROC-AUC:  {metrics['roc_auc']:.4f}")
                print(f"     CV Score: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
            else:
                print(f"  ❌ Failed to evaluate {name}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results available. Run model comparison first.")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            name: result['metrics'] for name, result in self.results.items()
        }).T
        
        # Sort by ROC-AUC score
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        print("\n📊 PERFORMANCE RANKING (by ROC-AUC):")
        print("-" * 50)
        print(f"{'Rank':<5} {'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 50)
        
        for i, (model_name, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i:<5} {model_name:<25} {row['accuracy']:<10.4f} {row['f1_score']:<10.4f} {row['roc_auc']:<10.4f}")
        
        # Best performing models
        best_model = results_df.index[0]
        second_best = results_df.index[1]
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
        print(f"🥈 SECOND BEST MODEL: {second_best}")
        
        # Detailed comparison of top 2 models
        print(f"\n🔍 DETAILED COMPARISON - TOP 2 MODELS:")
        print("-" * 40)
        
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        print(f"{'Metric':<15} {best_model:<15} {second_best:<15} {'Difference':<15}")
        print("-" * 60)
        
        for metric in comparison_metrics:
            best_val = results_df.loc[best_model, metric]
            second_val = results_df.loc[second_best, metric]
            diff = best_val - second_val
            print(f"{metric:<15} {best_val:<15.4f} {second_val:<15.4f} {diff:<15.4f}")
        
        # Model characteristics analysis
        print(f"\n📋 MODEL CHARACTERISTICS:")
        print("-" * 25)
        
        for model_name in [best_model, second_best]:
            print(f"\n{model_name}:")
            print(f"  Description: {self.results[model_name]['description']}")
            print(f"  Cross-validation: {results_df.loc[model_name, 'cv_mean']:.4f} ± {results_df.loc[model_name, 'cv_std']:.4f}")
        
        # Business suitability analysis
        print(f"\n💼 BUSINESS SUITABILITY ANALYSIS:")
        print("-" * 30)
        
        self._analyze_business_suitability(best_model, second_best, results_df)
        
        return results_df, best_model, second_best
    
    def _analyze_business_suitability(self, best_model, second_best, results_df):
        """Analyze business suitability of models"""
        
        # Interpretability scores (subjective ranking)
        interpretability = {
            'Logistic Regression': 9,
            'Decision Tree': 10,
            'Naive Bayes': 8,
            'K-Nearest Neighbors': 7,
            'Random Forest': 6,
            'Gradient Boosting': 4,
            'XGBoost': 3,
            'LightGBM': 3,
            'Support Vector Machine': 2
        }
        
        # Speed scores (subjective ranking)
        speed = {
            'Naive Bayes': 10,
            'Logistic Regression': 9,
            'K-Nearest Neighbors': 8,
            'Decision Tree': 8,
            'LightGBM': 7,
            'Random Forest': 6,
            'XGBoost': 5,
            'Gradient Boosting': 4,
            'Support Vector Machine': 3
        }
        
        models_to_analyze = [best_model, second_best]
        
        for model_name in models_to_analyze:
            print(f"\n{model_name}:")
            print(f"  • Performance Score: {results_df.loc[model_name, 'roc_auc']:.4f}")
            print(f"  • Interpretability: {interpretability.get(model_name, 5)}/10")
            print(f"  • Training Speed: {speed.get(model_name, 5)}/10")
            print(f"  • Suitable for: ", end="")
            
            # Business use case recommendations
            if model_name in ['Logistic Regression', 'Decision Tree']:
                print("Regulatory compliance, interpretable insights")
            elif model_name in ['Random Forest', 'Gradient Boosting']:
                print("Balanced performance and interpretability")
            elif model_name in ['XGBoost', 'LightGBM']:
                print("High-performance production systems")
            elif model_name == 'Support Vector Machine':
                print("Small datasets, complex decision boundaries")
            else:
                print("General purpose applications")
    
    def generate_visualizations(self):
        """Generate comparison visualizations"""
        if not self.results:
            print("No results to visualize")
            return
        
        print("\n📊 Generating model comparison visualizations...")
        
        # Prepare data for visualization
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        metric_data = {metric: [self.results[model]['metrics'][metric] for model in models] for metric in metrics}
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance comparison bar chart
        x_pos = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            ax1.bar(x_pos + i*width, metric_data[metric], width, label=metric.upper())
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x_pos + width * 2)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC-AUC ranking
        roc_scores = [self.results[model]['metrics']['roc_auc'] for model in models]
        sorted_indices = np.argsort(roc_scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [roc_scores[i] for i in sorted_indices]
        
        bars = ax2.barh(sorted_models, sorted_scores, color='skyblue')
        ax2.set_xlabel('ROC-AUC Score')
        ax2.set_title('Models Ranked by ROC-AUC')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, sorted_scores):
            ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center')
        
        # 3. Cross-validation scores
        cv_means = [self.results[model]['metrics']['cv_mean'] for model in models]
        cv_stds = [self.results[model]['metrics']['cv_std'] for model in models]
        
        ax3.errorbar(range(len(models)), cv_means, yerr=cv_stds, 
                    fmt='o-', capsize=5, capthick=2)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Cross-Validation Score')
        ax3.set_title('Cross-Validation Performance')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Radar chart for top 3 models
        top_3_models = sorted_models[:3]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        for model in top_3_models:
            values = [self.results[model]['metrics'][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax4.plot(angles, values, 'o-', linewidth=2, label=model)
            ax4.fill(angles, values, alpha=0.1)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels([m.upper() for m in metrics])
        ax4.set_title('Top 3 Models - Performance Radar')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to results/model_comparison.png")
    
    def run_complete_comparison(self):
        """Run the complete model comparison pipeline"""
        print("🚀 Starting Comprehensive Model Comparison")
        print("="*70)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return None
        
        # Initialize models
        self.initialize_models()
        
        # Compare models
        self.compare_all_models()
        
        # Generate report
        results_df, best_model, second_best = self.generate_comparison_report()
        
        # Generate visualizations
        self.generate_visualizations()
        
        print("\n" + "="*70)
        print("✅ MODEL COMPARISON COMPLETE!")
        print("="*70)
        print(f"\n🏆 Recommended model: {best_model}")
        print(f"🥈 Alternative model: {second_best}")
        print("📊 Detailed results and visualizations saved")
        
        return {
            'results_df': results_df,
            'best_model': best_model,
            'second_best': second_best,
            'all_results': self.results
        }

def main():
    """Main function to run model comparison"""
    # Initialize comparator
    comparator = MLModelComparator('data/raw/online_retail_customer_churn.csv')
    
    # Run complete comparison
    comparison_results = comparator.run_complete_comparison()
    
    if comparison_results:
        print("\n🎯 Model comparison results available for assignment!")
        return comparator, comparison_results
    else:
        print("\n❌ Please ensure dataset is available in data/raw/")
        return None, None

if __name__ == "__main__":
    comparator, results = main()
