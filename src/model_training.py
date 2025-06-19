"""
ML Model Training, Testing and Validation Module
BUS8405 Assignment - CLO3: Model Development and Performance Evaluation
Implementation: Complete training pipeline with hyperparameter optimization and validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
                           classification_report, matthews_corrcoef, log_loss)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    """
    Comprehensive model training and evaluation for churn prediction
    """
    
    def __init__(self, data_path, model_choice='RandomForest'):
        """
        Initialize model trainer
        
        Args:
            data_path (str): Path to the dataset
            model_choice (str): Choice of model to train
        """
        self.data_path = data_path
        self.model_choice = model_choice
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.evaluation_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with comprehensive preprocessing"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            
            self._comprehensive_preprocessing()
            self._create_train_val_test_split()
            
            return True
        except FileNotFoundError:
            print("❌ Dataset not found. Please ensure the dataset is in data/raw/")
            return False
    
    def _comprehensive_preprocessing(self):
        """Comprehensive data preprocessing"""
        print("\n" + "="*60)
        print("COMPREHENSIVE DATA PREPROCESSING")
        print("="*60)
        
        # Create copy for processing
        df_processed = self.df.copy()
        
        # Remove customer ID if present
        if 'Customer_ID' in df_processed.columns:
            df_processed = df_processed.drop('Customer_ID', axis=1)
            print("✓ Removed Customer_ID column")
        
        # Handle missing values (if any)
        missing_before = df_processed.isnull().sum().sum()
        if missing_before > 0:
            print(f"Handling {missing_before} missing values...")
            # Fill numerical columns with median
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'Target_Churn' and df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        print(f"✓ Missing values after preprocessing: {df_processed.isnull().sum().sum()}")
        
        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        if 'Target_Churn' in categorical_columns:
            categorical_columns.remove('Target_Churn')
        
        print(f"✓ Encoding categorical variables: {categorical_columns}")
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle target variable
        if 'Target_Churn' in df_processed.columns:
            if df_processed['Target_Churn'].dtype == 'object' or df_processed['Target_Churn'].dtype == 'bool':
                df_processed['Target_Churn'] = df_processed['Target_Churn'].astype(str)
                df_processed['Target_Churn'] = (df_processed['Target_Churn'] == 'True').astype(int)
        
        # Feature engineering
        self._feature_engineering(df_processed)
        
        # Separate features and target
        self.y = df_processed['Target_Churn']
        self.X = df_processed.drop('Target_Churn', axis=1)
        
        print(f"✓ Final feature set: {self.X.shape[1]} features")
        print(f"✓ Target distribution: {self.y.value_counts().to_dict()}")
        
    def _feature_engineering(self, df):
        """Create additional features for better model performance"""
        print("\n📈 Feature Engineering:")
        
        # Calculate customer value score
        if all(col in df.columns for col in ['Total_Spend', 'Years_as_Customer']):
            df['Customer_Value_Score'] = df['Total_Spend'] / (df['Years_as_Customer'] + 1)
            print("✓ Created Customer_Value_Score")
        
        # Calculate purchase frequency
        if all(col in df.columns for col in ['Num_of_Purchases', 'Years_as_Customer']):
            df['Purchase_Frequency'] = df['Num_of_Purchases'] / (df['Years_as_Customer'] + 1)
            print("✓ Created Purchase_Frequency")
        
        # Calculate return rate
        if all(col in df.columns for col in ['Num_of_Returns', 'Num_of_Purchases']):
            df['Return_Rate'] = df['Num_of_Returns'] / (df['Num_of_Purchases'] + 1)
            print("✓ Created Return_Rate")
        
        # Calculate support contact rate
        if all(col in df.columns for col in ['Num_of_Support_Contacts', 'Num_of_Purchases']):
            df['Support_Contact_Rate'] = df['Num_of_Support_Contacts'] / (df['Num_of_Purchases'] + 1)
            print("✓ Created Support_Contact_Rate")
        
        # Age groups
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Elder'])
            df['Age_Group'] = df['Age_Group'].cat.codes
            print("✓ Created Age_Group")
        
        # Income categories
        if 'Annual_Income' in df.columns:
            df['Income_Category'] = pd.qcut(df['Annual_Income'], q=3, 
                                          labels=['Low', 'Medium', 'High'])
            df['Income_Category'] = df['Income_Category'].cat.codes
            print("✓ Created Income_Category")
    
    def _create_train_val_test_split(self):
        """Create train/validation/test splits"""
        # First split: train+val vs test (80-20)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Second split: train vs val (75-25 of remaining 80%)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"\n📊 Data Split Summary:")
        print(f"Training set:   {self.X_train.shape[0]} samples ({self.X_train.shape[0]/len(self.X)*100:.1f}%)")
        print(f"Validation set: {self.X_val.shape[0]} samples ({self.X_val.shape[0]/len(self.X)*100:.1f}%)")
        print(f"Test set:       {self.X_test.shape[0]} samples ({self.X_test.shape[0]/len(self.X)*100:.1f}%)")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def select_and_configure_model(self):
        """Select and configure the chosen model"""
        print(f"\n🔧 Configuring {self.model_choice} model...")
        
        if self.model_choice == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1            )
            self.use_scaled_data = False
            
        elif self.model_choice == 'LogisticRegression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
            self.use_scaled_data = True
            
        elif self.model_choice == 'XGBoost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            self.use_scaled_data = False
            
        elif self.model_choice == 'GradientBoosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            self.use_scaled_data = False
            
        elif self.model_choice == 'K-Nearest Neighbors' or self.model_choice == 'KNeighbors':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='auto',
                metric='minkowski'
            )
            self.use_scaled_data = True
            
        elif self.model_choice == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            self.use_scaled_data = False
            
        elif self.model_choice == 'SVM' or self.model_choice == 'Support Vector Machine':
            from sklearn.svm import SVC
            self.model = SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
            self.use_scaled_data = True
            
        else:
            raise ValueError(f"Model choice '{self.model_choice}' not supported")
        
        print(f"✓ Model configured: {type(self.model).__name__}")
        print(f"✓ Using scaled data: {self.use_scaled_data}")
    
    def hyperparameter_tuning(self, method='grid', cv_folds=5):
        """Perform hyperparameter tuning"""
        print(f"\n🎯 Hyperparameter Tuning using {method.upper()} search...")
        
        # Choose appropriate data
        X_train = self.X_train_scaled if self.use_scaled_data else self.X_train
        
        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'DecisionTree': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'Support Vector Machine': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        param_grid = param_grids.get(self.model_choice, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {self.model_choice}")
            return
        
        # Perform search
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(
                self.model, param_grid, cv=cv, 
                scoring='roc_auc', n_jobs=-1, verbose=1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                self.model, param_grid, cv=cv, 
                scoring='roc_auc', n_iter=20, n_jobs=-1, 
                random_state=42, verbose=1
            )
        
        search.fit(X_train, self.y_train)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        
        print(f"✅ Best parameters found:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"✅ Best CV score: {search.best_score_:.4f}")
        
        return search.best_params_, search.best_score_
    
    def train_model(self):
        """Train the model with comprehensive monitoring"""
        print(f"\n🚀 Training {self.model_choice} model...")
        
        # Choose appropriate data
        X_train = self.X_train_scaled if self.use_scaled_data else self.X_train
        X_val = self.X_val_scaled if self.use_scaled_data else self.X_val
        
        # Train model
        if hasattr(self.model, 'fit') and 'XGB' in str(type(self.model)):
            # XGBoost with validation monitoring
            self.model.fit(
                X_train, self.y_train,
                eval_set=[(X_val, self.y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, self.y_train)
        
        print("✅ Model training completed!")
        
        # Save the trained model
        model_filename = f"models/{self.model_choice.lower()}_churn_model.joblib"
        joblib.dump(self.model, model_filename)
        print(f"✅ Model saved to {model_filename}")
    
    def comprehensive_evaluation(self):
        """Comprehensive model evaluation on all datasets"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        datasets = {
            'Training': (self.X_train_scaled if self.use_scaled_data else self.X_train, self.y_train),
            'Validation': (self.X_val_scaled if self.use_scaled_data else self.X_val, self.y_val),
            'Test': (self.X_test_scaled if self.use_scaled_data else self.X_test, self.y_test)
        }
        
        self.evaluation_results = {}
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\n📊 Evaluating on {dataset_name} set...")
            
            # Predictions
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            # Calculate all metrics
            metrics = self._calculate_all_metrics(y, y_pred, y_pred_proba)
            
            self.evaluation_results[dataset_name] = {
                'y_true': y,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': metrics
            }
            
            # Print key metrics
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        
        # Check for overfitting
        self._check_overfitting()
        
        return self.evaluation_results
    
    def _calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive set of evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            
            # Class-specific metrics
            'precision_class0': precision_score(y_true, y_pred, pos_label=0),
            'precision_class1': precision_score(y_true, y_pred, pos_label=1),
            'recall_class0': recall_score(y_true, y_pred, pos_label=0),
            'recall_class1': recall_score(y_true, y_pred, pos_label=1),
            'f1_class0': f1_score(y_true, y_pred, pos_label=0),
            'f1_class1': f1_score(y_true, y_pred, pos_label=1),
        }
        
        return metrics
    
    def _check_overfitting(self):
        """Check for overfitting by comparing train vs validation performance"""
        print(f"\n🔍 Overfitting Analysis:")
        print("-" * 25)
        
        train_metrics = self.evaluation_results['Training']['metrics']
        val_metrics = self.evaluation_results['Validation']['metrics']
        
        key_metrics = ['accuracy', 'f1_score', 'roc_auc']
        
        for metric in key_metrics:
            train_score = train_metrics[metric]
            val_score = val_metrics[metric]
            diff = train_score - val_score
            
            if diff > 0.1:
                status = "🔴 Possible overfitting"
            elif diff > 0.05:
                status = "🟡 Slight overfitting"
            else:
                status = "🟢 Good generalization"
            
            print(f"  {metric.upper()}:")
            print(f"    Train: {train_score:.4f}, Val: {val_score:.4f}, Diff: {diff:.4f} - {status}")
    
    def generate_detailed_report(self):
        """Generate detailed evaluation report"""
        print("\n" + "="*80)
        print("DETAILED EVALUATION REPORT")
        print("="*80)
        
        # Test set detailed analysis
        test_results = self.evaluation_results['Test']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        y_pred_proba = test_results['y_pred_proba']
        
        print("\n📋 CLASSIFICATION REPORT:")
        print("-" * 30)
        print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
        
        print("\n📊 CONFUSION MATRIX:")
        print("-" * 20)
        cm = confusion_matrix(y_true, y_pred)
        print(f"                 Predicted")
        print(f"Actual    No Churn    Churn")
        print(f"No Churn    {cm[0,0]:6d}    {cm[0,1]:5d}")
        print(f"Churn       {cm[1,0]:6d}    {cm[1,1]:5d}")
        
        # Calculate business metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        print(f"\n💼 BUSINESS METRICS:")
        print(f"  True Negatives:  {tn:4d} (correctly identified non-churners)")
        print(f"  False Positives: {fp:4d} (incorrectly flagged as churners)")
        print(f"  False Negatives: {fn:4d} (missed churners)")
        print(f"  True Positives:  {tp:4d} (correctly identified churners)")
        print(f"  Specificity:     {specificity:.4f} (correctly identifying non-churners)")
        print(f"  Sensitivity:     {sensitivity:.4f} (correctly identifying churners)")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self._analyze_feature_importance()
        
        return {
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'business_metrics': {
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'specificity': specificity,
                'sensitivity': sensitivity
            }
        }
    
    def _analyze_feature_importance(self):
        """Analyze and display feature importance"""
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS:")
        print("-" * 35)
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        return feature_importance
    
    def generate_visualizations(self):
        """Generate comprehensive evaluation visualizations"""
        print("\n📊 Generating evaluation visualizations...")
        
        test_results = self.evaluation_results['Test']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        y_pred_proba = test_results['y_pred_proba']
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = plt.subplot(3, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(3, 3, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # 4. Prediction Probability Distribution
        ax4 = plt.subplot(3, 3, 4)
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='No Churn', color='blue')
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Churn', color='red')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # 5. Calibration Plot
        ax5 = plt.subplot(3, 3, 5)
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        # 6. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            ax6 = plt.subplot(3, 3, 6)
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Plot top 10 features
            top_features = feature_importance.tail(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance')
        
        # 7. Performance Comparison Across Sets
        ax7 = plt.subplot(3, 3, 7)
        datasets = ['Training', 'Validation', 'Test']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            values = [self.evaluation_results[dataset]['metrics'][metric] for dataset in datasets]
            plt.plot(datasets, values, marker='o', label=metric.upper())
        
        plt.ylabel('Score')
        plt.title('Performance Across Datasets')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 8. Error Analysis
        ax8 = plt.subplot(3, 3, 8)
        errors = np.abs(y_true - y_pred_proba)
        plt.hist(errors, bins=30, alpha=0.7, color='orange')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        
        # 9. Residual Plot
        ax9 = plt.subplot(3, 3, 9)
        residuals = y_true - y_pred_proba
        plt.scatter(y_pred_proba, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig(f'results/{self.model_choice.lower()}_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Evaluation visualizations saved to results/{self.model_choice.lower()}_evaluation.png")
    
    def justify_metric_selection(self):
        """Provide justification for metric selection in business context"""
        print("\n" + "="*60)
        print("METRIC SELECTION JUSTIFICATION")
        print("="*60)
        
        print("\n🎯 BUSINESS CONTEXT: Customer Churn Prediction")
        print("-" * 45)
        
        metrics_analysis = {
            'ROC-AUC': {
                'value': self.evaluation_results['Test']['metrics']['roc_auc'],
                'importance': 'HIGH',
                'justification': 'Measures model ability to distinguish between churners and non-churners across all thresholds'
            },
            'Precision (Churn)': {
                'value': self.evaluation_results['Test']['metrics']['precision_class1'],
                'importance': 'HIGH',
                'justification': 'Critical for targeted marketing - reduces wasted resources on false positives'
            },
            'Recall (Churn)': {
                'value': self.evaluation_results['Test']['metrics']['recall_class1'],
                'importance': 'VERY HIGH',
                'justification': 'Essential for revenue protection - missing actual churners is costly'
            },
            'F1-Score': {
                'value': self.evaluation_results['Test']['metrics']['f1_score'],
                'importance': 'HIGH',
                'justification': 'Balances precision and recall, good overall performance indicator'
            },
            'Accuracy': {
                'value': self.evaluation_results['Test']['metrics']['accuracy'],
                'importance': 'MEDIUM',
                'justification': 'General performance but can be misleading with imbalanced classes'
            }
        }
        
        print(f"\n📊 METRIC ANALYSIS:")
        for metric, info in metrics_analysis.items():
            print(f"\n{metric}:")
            print(f"  Value: {info['value']:.4f}")
            print(f"  Business Importance: {info['importance']}")
            print(f"  Justification: {info['justification']}")
        
        # Recommend primary metric
        recall_churn = self.evaluation_results['Test']['metrics']['recall_class1']
        roc_auc = self.evaluation_results['Test']['metrics']['roc_auc']
        
        print(f"\n🏆 RECOMMENDED PRIMARY METRIC:")
        if recall_churn >= 0.8:
            print("RECALL (for Churn class)")
            print("Reasoning: High recall ensures we catch most churners, crucial for retention")
        else:
            print("ROC-AUC")
            print("Reasoning: Provides balanced view of model performance across all thresholds")
        
        print(f"\n💡 BUSINESS IMPLICATIONS:")
        print("• High Recall: Minimize missed churn opportunities (false negatives)")
        print("• High Precision: Reduce unnecessary retention costs (false positives)")
        print("• High ROC-AUC: Overall model quality for ranking customers by churn risk")
        
        return metrics_analysis
    
    def run_complete_training_pipeline(self, tune_hyperparameters=True):
        """Run the complete training and evaluation pipeline"""
        print("🚀 Starting Complete Model Training Pipeline")
        print("="*70)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return None
        
        # Select and configure model
        self.select_and_configure_model()
        
        # Hyperparameter tuning (optional)
        if tune_hyperparameters:
            try:
                best_params, best_score = self.hyperparameter_tuning()
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}")
                print("Proceeding with default parameters...")
        
        # Train model
        self.train_model()
        
        # Comprehensive evaluation
        evaluation_results = self.comprehensive_evaluation()
        
        # Generate detailed report
        detailed_report = self.generate_detailed_report()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Justify metric selection
        metric_justification = self.justify_metric_selection()
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n🎯 Model: {self.model_choice}")
        print(f"🎯 Test ROC-AUC: {evaluation_results['Test']['metrics']['roc_auc']:.4f}")
        print(f"🎯 Test F1-Score: {evaluation_results['Test']['metrics']['f1_score']:.4f}")
        print("📊 Detailed results and visualizations saved")
        
        return {
            'model': self.model,
            'evaluation_results': evaluation_results,
            'detailed_report': detailed_report,
            'metric_justification': metric_justification,
            'model_path': f"models/{self.model_choice.lower()}_churn_model.joblib"
        }

def main(model_choice='RandomForest', tune_hyperparameters=True):
    """Main function to run model training"""
    # Initialize trainer
    trainer = ChurnModelTrainer('data/raw/online_retail_customer_churn.csv', model_choice)
    
    # Run complete training pipeline
    training_results = trainer.run_complete_training_pipeline(tune_hyperparameters)
    
    if training_results:
        print(f"\n🎯 {model_choice} training results available for assignment!")
        return trainer, training_results
    else:
        print("\n❌ Please ensure dataset is available in data/raw/")
        return None, None

if __name__ == "__main__":
    trainer, results = main()
