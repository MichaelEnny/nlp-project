#!/usr/bin/env python3
"""
Text Classification with Messy Data

This script implements a complete text classification solution for messy social media text.
It handles emojis, random capitalization, extra spaces, slang words, and other messiness patterns.

Assignment: Text Classification
- Categories: sports, politics, tech, food, entertainment
- Challenge: Handle messy text data effectively
- Goal: Build robust classification model with performance analysis

Author: Claude Code AI Assistant
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter
import string
import time
from datetime import datetime

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Download required NLTK data
def setup_nltk():
    """Download required NLTK data if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

# Configuration
plt.style.use('default')
warnings.filterwarnings('ignore')
np.random.seed(42)

class TextClassificationPipeline:
    """Complete text classification pipeline for messy data"""

    def __init__(self, dataset_path='dataset/text classifcation.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.df_processed = None
        self.text_col = None
        self.target_col = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.vectorizers = {}
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_features = None
        self.best_accuracy = 0

        print("="*60)
        print("TEXT CLASSIFICATION WITH MESSY DATA")
        print("="*60)
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def load_and_explore_data(self):
        """Load dataset and perform exploratory data analysis"""
        print("1. LOADING AND EXPLORING DATA")
        print("-" * 40)

        # Load dataset
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded successfully: {self.df.shape}")
        except FileNotFoundError:
            print(f"❌ Error: Dataset not found at {self.dataset_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False

        # Identify columns
        self.text_col = self.df.columns[0]
        self.target_col = self.df.columns[1]

        print(f"📝 Text column: '{self.text_col}'")
        print(f"🎯 Target column: '{self.target_col}'")
        print(f"💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Basic statistics
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total samples: {len(self.df):,}")
        print(f"   • Unique texts: {self.df[self.text_col].nunique():,}")
        print(f"   • Missing values: {self.df.isnull().sum().sum()}")

        # Target distribution
        target_counts = self.df[self.target_col].value_counts()
        print(f"\n🎯 Target Distribution:")
        for category, count in target_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   • {category}: {count:,} ({percentage:.1f}%)")

        # Class balance check
        balance_ratio = target_counts.min() / target_counts.max()
        print(f"\n⚖️  Class balance ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.5:
            print("   ⚠️  Dataset shows class imbalance")
        else:
            print("   ✅ Dataset is reasonably balanced")

        # Text statistics
        texts = self.df[self.text_col].astype(str)
        text_lengths = texts.str.len()
        word_counts = texts.str.split().str.len()

        print(f"\n📏 Text Statistics:")
        print(f"   • Avg length: {text_lengths.mean():.1f} chars ({word_counts.mean():.1f} words)")
        print(f"   • Length range: {text_lengths.min()}-{text_lengths.max()} chars")
        print(f"   • Word range: {word_counts.min()}-{word_counts.max()} words")

        return True

    def analyze_messiness_patterns(self):
        """Analyze messiness patterns in the text data"""
        print("\n2. ANALYZING MESSINESS PATTERNS")
        print("-" * 40)

        texts = self.df[self.text_col].astype(str)

        # Define pattern detection functions
        patterns = {
            'emojis': texts.str.contains(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff]', regex=True),
            'extra_spaces': texts.str.contains(r'\s{2,}', regex=True),
            'mixed_case': texts.str.contains(r'[a-z]', regex=True) & texts.str.contains(r'[A-Z]', regex=True),
            'all_caps_words': texts.str.contains(r'\b[A-Z]{2,}\b', regex=True),
            'numbers': texts.str.contains(r'\d', regex=True),
            'special_chars': texts.str.contains(r'[^a-zA-Z0-9\s]', regex=True),
            'repeated_chars': texts.str.contains(r'(.)\1{2,}', regex=True),
            'urls': texts.str.contains(r'http[s]?://|www\.', regex=True, case=False),
            'hashtags': texts.str.contains(r'#\w+', regex=True),
            'mentions': texts.str.contains(r'@\w+', regex=True)
        }

        print("🔍 Messiness Pattern Analysis:")
        for pattern_name, pattern_mask in patterns.items():
            count = pattern_mask.sum()
            percentage = pattern_mask.mean() * 100
            print(f"   • {pattern_name.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")

        # Overall messiness
        overall_messiness = sum(patterns.values()) > 0
        messy_count = overall_messiness.sum()
        messy_percentage = overall_messiness.mean() * 100
        print(f"\n📈 Overall Messiness: {messy_count:,} texts ({messy_percentage:.1f}%) contain messiness patterns")

        # Show examples by category
        print(f"\n📋 Sample Messy Texts by Category:")
        for category in self.df[self.target_col].unique():
            category_data = self.df[self.df[self.target_col] == category]
            sample_text = category_data[self.text_col].iloc[0]
            print(f"   • {category}: {sample_text[:80]}{'...' if len(sample_text) > 80 else ''}")

        return patterns

    def create_preprocessing_pipeline(self):
        """Create and configure text preprocessing pipeline"""
        print("\n3. CREATING PREPROCESSING PIPELINE")
        print("-" * 40)

        class TextPreprocessor:
            def __init__(self,
                         remove_emojis=True,
                         normalize_whitespace=True,
                         lowercase=True,
                         remove_special_chars=True,
                         remove_numbers=False,
                         remove_stopwords=True,
                         lemmatize=True,
                         min_word_length=2):

                self.remove_emojis = remove_emojis
                self.normalize_whitespace = normalize_whitespace
                self.lowercase = lowercase
                self.remove_special_chars = remove_special_chars
                self.remove_numbers = remove_numbers
                self.remove_stopwords = remove_stopwords
                self.lemmatize = lemmatize
                self.min_word_length = min_word_length

                # Initialize NLTK components
                if self.remove_stopwords:
                    self.stop_words = set(stopwords.words('english'))

                if self.lemmatize:
                    self.lemmatizer = WordNetLemmatizer()

            def clean_text(self, text):
                """Apply all cleaning steps to a single text"""
                if pd.isna(text) or text == '':
                    return ''

                text = str(text)

                # Remove emojis
                if self.remove_emojis:
                    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff]'
                    text = re.sub(emoji_pattern, ' ', text)

                # Normalize whitespace
                if self.normalize_whitespace:
                    text = re.sub(r'\s+', ' ', text).strip()

                # Convert to lowercase
                if self.lowercase:
                    text = text.lower()

                # Remove repeated characters (more than 2 consecutive)
                text = re.sub(r'(.)\1{2,}', r'\1\1', text)

                # Remove special characters and punctuation
                if self.remove_special_chars:
                    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

                # Remove numbers if specified
                if self.remove_numbers:
                    text = re.sub(r'\d+', ' ', text)

                # Tokenization and further processing
                try:
                    words = word_tokenize(text)
                except:
                    words = text.split()

                # Remove stopwords
                if self.remove_stopwords:
                    words = [word for word in words if word.lower() not in self.stop_words]

                # Remove short words
                words = [word for word in words if len(word) >= self.min_word_length]

                # Lemmatization
                if self.lemmatize:
                    try:
                        words = [self.lemmatizer.lemmatize(word) for word in words]
                    except:
                        pass  # Skip lemmatization if it fails

                return ' '.join(words)

            def fit_transform(self, texts):
                """Apply preprocessing to a series of texts"""
                return texts.apply(self.clean_text)

        self.preprocessor = TextPreprocessor()

        print("🛠️  Text Preprocessing Pipeline Created:")
        print("   ✅ Remove emojis")
        print("   ✅ Normalize whitespace")
        print("   ✅ Convert to lowercase")
        print("   ✅ Remove special characters")
        print("   ✅ Fix repeated characters")
        print("   ✅ Remove stopwords")
        print("   ✅ Lemmatization")
        print("   ✅ Filter short words")

        return True

    def preprocess_texts(self):
        """Apply preprocessing to the dataset"""
        print("\n4. PREPROCESSING TEXTS")
        print("-" * 40)

        # Create a copy for preprocessing
        self.df_processed = self.df.copy()

        print("🔄 Applying preprocessing to texts...")
        start_time = time.time()

        # Apply preprocessing
        self.df_processed['cleaned_text'] = self.preprocessor.fit_transform(self.df[self.text_col])

        # Remove empty texts after cleaning
        initial_count = len(self.df_processed)
        self.df_processed = self.df_processed[self.df_processed['cleaned_text'].str.strip() != '']
        final_count = len(self.df_processed)

        processing_time = time.time() - start_time

        print(f"✅ Preprocessing completed in {processing_time:.2f} seconds")
        print(f"   • Texts processed: {initial_count:,}")
        print(f"   • Texts after cleaning: {final_count:,}")
        print(f"   • Texts removed (empty): {initial_count - final_count:,}")

        # Compare statistics
        original_lengths = self.df[self.text_col].astype(str).str.len()
        cleaned_lengths = self.df_processed['cleaned_text'].str.len()
        original_words = self.df[self.text_col].astype(str).str.split().str.len()
        cleaned_words = self.df_processed['cleaned_text'].str.split().str.len()

        print(f"\n📊 Preprocessing Impact:")
        print(f"   • Avg char length: {original_lengths.mean():.1f} → {cleaned_lengths.mean():.1f} ({((cleaned_lengths.mean() - original_lengths.mean()) / original_lengths.mean() * 100):+.1f}%)")
        print(f"   • Avg word count: {original_words.mean():.1f} → {cleaned_words.mean():.1f} ({((cleaned_words.mean() - original_words.mean()) / original_words.mean() * 100):+.1f}%)")

        # Show examples
        print(f"\n📝 Preprocessing Examples:")
        for i in range(min(3, len(self.df_processed))):
            original = str(self.df.iloc[i][self.text_col])
            cleaned = self.df_processed.iloc[i]['cleaned_text']
            category = self.df_processed.iloc[i][self.target_col]

            print(f"   {i+1}. Category: {category}")
            print(f"      Original: {original[:60]}{'...' if len(original) > 60 else ''}")
            print(f"      Cleaned:  {cleaned[:60]}{'...' if len(cleaned) > 60 else ''}")
            print()

        return True

    def engineer_features(self):
        """Create multiple feature representations"""
        print("5. FEATURE ENGINEERING")
        print("-" * 40)

        # Prepare data for modeling
        X = self.df_processed['cleaned_text']
        y = self.df_processed[self.target_col]

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"🎯 Target encoding:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"   • {class_name} → {i}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        print(f"\n📊 Data Split:")
        print(f"   • Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   • Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Create feature representations
        print(f"\n🔧 Creating Feature Representations:")

        # 1. Bag of Words
        print("   1. Bag of Words...")
        bow_vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_bow = bow_vectorizer.fit_transform(X_train)
        X_test_bow = bow_vectorizer.transform(X_test)

        # 2. TF-IDF
        print("   2. TF-IDF...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # 3. Character-level TF-IDF
        print("   3. Character-level TF-IDF...")
        char_tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=3000,
            min_df=2,
            max_df=0.95
        )
        X_train_char = char_tfidf_vectorizer.fit_transform(X_train)
        X_test_char = char_tfidf_vectorizer.transform(X_test)

        # Store vectorizers and feature sets
        self.vectorizers = {
            'bow': bow_vectorizer,
            'tfidf': tfidf_vectorizer,
            'char_tfidf': char_tfidf_vectorizer
        }

        self.feature_sets = {
            'Bag of Words': (X_train_bow, X_test_bow),
            'TF-IDF': (X_train_tfidf, X_test_tfidf),
            'Character TF-IDF': (X_train_char, X_test_char)
        }

        self.data_splits = (X_train, X_test, y_train, y_test)

        print(f"\n✅ Feature Engineering Completed:")
        print(f"   • BOW features: {X_train_bow.shape[1]:,} (sparsity: {(1 - X_train_bow.nnz / (X_train_bow.shape[0] * X_train_bow.shape[1]))*100:.1f}%)")
        print(f"   • TF-IDF features: {X_train_tfidf.shape[1]:,} (sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.1f}%)")
        print(f"   • Char TF-IDF features: {X_train_char.shape[1]:,} (sparsity: {(1 - X_train_char.nnz / (X_train_char.shape[0] * X_train_char.shape[1]))*100:.1f}%)")

        return True

    def train_and_evaluate_models(self):
        """Train and evaluate multiple classification models"""
        print("\n6. MODEL TRAINING AND EVALUATION")
        print("-" * 40)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Multinomial Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM Linear': SVC(kernel='linear', random_state=42, probability=True)
        }

        print(f"🤖 Training {len(models)} models with {len(self.feature_sets)} feature types...")
        print(f"   Total combinations: {len(models) * len(self.feature_sets)}")

        X_train, X_test, y_train, y_test = self.data_splits
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.results = []
        trained_models = {}

        for feature_name, (X_train_feat, X_test_feat) in self.feature_sets.items():
            print(f"\n   📊 {feature_name} Features:")

            for model_name, model in models.items():
                print(f"      🔄 {model_name}...", end=" ")

                start_time = time.time()

                # Cross-validation
                cv_scores = cross_val_score(model, X_train_feat, y_train, cv=cv, scoring='accuracy')

                # Train on full training set
                model.fit(X_train_feat, y_train)

                # Test set evaluation
                y_pred = model.predict(X_test_feat)
                test_accuracy = accuracy_score(y_test, y_pred)

                training_time = time.time() - start_time

                # Store results
                result = {
                    'Model': model_name,
                    'Features': feature_name,
                    'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std(),
                    'Test_Accuracy': test_accuracy,
                    'Training_Time': training_time,
                    'Predictions': y_pred
                }
                self.results.append(result)

                # Store trained model
                model_key = f"{model_name}_{feature_name}"
                trained_models[model_key] = model

                print(f"Acc: {test_accuracy:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} | Time: {training_time:.1f}s")

        self.trained_models = trained_models

        # Find best model
        results_df = pd.DataFrame(self.results)
        best_result = results_df.loc[results_df['Test_Accuracy'].idxmax()]

        self.best_model = trained_models[f"{best_result['Model']}_{best_result['Features']}"]
        self.best_features = best_result['Features']
        self.best_accuracy = best_result['Test_Accuracy']

        print(f"\n🏆 BEST MODEL: {best_result['Model']} with {best_result['Features']}")
        print(f"   • Test Accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        print(f"   • CV Accuracy: {best_result['CV_Mean']:.4f} ± {best_result['CV_Std']:.4f}")
        print(f"   • Training Time: {best_result['Training_Time']:.2f} seconds")

        return results_df, best_result, y_test

    def analyze_messiness_impact(self):
        """Analyze the impact of text messiness on classification performance"""
        print("\n7. MESSINESS IMPACT ANALYSIS")
        print("-" * 40)

        print("🔄 Training model on original messy text for comparison...")

        # Use original text without preprocessing
        X_original = self.df[self.text_col].astype(str)
        y_original = self.label_encoder.transform(self.df[self.target_col])

        # Split original data
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_original, y_original,
            test_size=0.2,
            random_state=42,
            stratify=y_original
        )

        # Create TF-IDF features for original text
        original_tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        X_orig_train_tfidf = original_tfidf.fit_transform(X_orig_train)
        X_orig_test_tfidf = original_tfidf.transform(X_orig_test)

        # Train model on original messy text
        messy_model = LogisticRegression(random_state=42, max_iter=1000)
        messy_model.fit(X_orig_train_tfidf, y_orig_train)
        messy_predictions = messy_model.predict(X_orig_test_tfidf)
        messy_accuracy = accuracy_score(y_orig_test, messy_predictions)

        print(f"✅ Messy text model trained")

        # Compare performance
        improvement = self.best_accuracy - messy_accuracy
        improvement_pct = (improvement / messy_accuracy) * 100

        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   • Messy text accuracy:   {messy_accuracy:.4f} ({messy_accuracy*100:.2f}%)")
        print(f"   • Clean text accuracy:   {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        print(f"   • Improvement:           {improvement:.4f} ({improvement_pct:+.2f}%)")

        if improvement > 0.02:
            impact_level = "SIGNIFICANT"
            recommendation = "essential"
        elif improvement > 0.005:
            impact_level = "MODERATE"
            recommendation = "beneficial"
        else:
            impact_level = "MINIMAL"
            recommendation = "optional"

        print(f"   • Impact level:          {impact_level}")
        print(f"   • Preprocessing is:      {recommendation}")

        # Vocabulary analysis
        messy_vocab_size = len(original_tfidf.vocabulary_)
        clean_vocab_size = len(self.vectorizers['tfidf'].vocabulary_)
        vocab_reduction = messy_vocab_size - clean_vocab_size
        vocab_reduction_pct = (vocab_reduction / messy_vocab_size) * 100

        print(f"\n📚 VOCABULARY ANALYSIS:")
        print(f"   • Messy text vocabulary:  {messy_vocab_size:,}")
        print(f"   • Clean text vocabulary:  {clean_vocab_size:,}")
        print(f"   • Vocabulary reduction:   {vocab_reduction:,} ({vocab_reduction_pct:.1f}%)")

        return {
            'messy_accuracy': messy_accuracy,
            'clean_accuracy': self.best_accuracy,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'impact_level': impact_level,
            'recommendation': recommendation,
            'vocab_reduction': vocab_reduction_pct
        }

    def generate_final_report(self, results_df, best_result, y_test, impact_analysis):
        """Generate comprehensive final report"""
        print("\n8. FINAL REPORT AND CONCLUSIONS")
        print("=" * 60)

        # Classification report for best model
        best_predictions = best_result['Predictions']
        class_names = self.label_encoder.classes_

        print(f"\n📊 DETAILED CLASSIFICATION REPORT")
        print(f"Model: {best_result['Model']} with {best_result['Features']}")
        print("-" * 50)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, best_predictions)

        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")

        # Overall metrics
        macro_avg_precision = np.mean(precision)
        macro_avg_recall = np.mean(recall)
        macro_avg_f1 = np.mean(f1)

        print("-" * 60)
        print(f"{'Macro Avg':<15} {macro_avg_precision:<10.4f} {macro_avg_recall:<10.4f} {macro_avg_f1:<10.4f} {support.sum():<10}")
        print(f"{'Accuracy':<15} {'':<10} {'':<10} {self.best_accuracy:<10.4f} {support.sum():<10}")

        # Model performance ranking
        print(f"\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 50)
        results_sorted = results_df.sort_values('Test_Accuracy', ascending=False)

        print(f"{'Rank':<4} {'Model':<25} {'Features':<18} {'Test Acc':<10} {'CV Acc':<12}")
        print("-" * 75)
        for rank, (_, row) in enumerate(results_sorted.iterrows(), 1):
            print(f"{rank:<4} {row['Model']:<25} {row['Features']:<18} {row['Test_Accuracy']:<10.4f} {row['CV_Mean']:<12.4f}")

        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Dataset: {len(self.df):,} samples across {len(class_names)} categories")
        print(f"   • Best algorithm: {best_result['Model']}")
        print(f"   • Best features: {best_result['Features']}")
        print(f"   • Final accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        print(f"   • Preprocessing impact: {impact_analysis['improvement_pct']:+.2f}% improvement")
        print(f"   • Impact level: {impact_analysis['impact_level']}")
        print(f"   • Vocabulary reduction: {impact_analysis['vocab_reduction']:.1f}%")

        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   1. Use {best_result['Model']} with {best_result['Features']} for production")
        print(f"   2. Text preprocessing is {impact_analysis['recommendation']} for this dataset")
        print(f"   3. Focus cleaning on: emojis, extra spaces, special characters")
        print(f"   4. Monitor performance on new messy text data")
        print(f"   5. Consider ensemble methods for potential improvement")

        # Final conclusion
        print(f"\n🎉 CONCLUSION:")
        if impact_analysis['improvement'] > 0.02:
            conclusion = f"Text preprocessing significantly improves classification performance by {impact_analysis['improvement_pct']:.1f}%. The messiness substantially impacts model accuracy, making preprocessing critical."
        elif impact_analysis['improvement'] > 0.005:
            conclusion = f"Text preprocessing moderately improves classification performance by {impact_analysis['improvement_pct']:.1f}%. The impact is noticeable but the model shows some robustness."
        else:
            conclusion = f"Text preprocessing has minimal impact ({impact_analysis['improvement_pct']:+.1f}%). The model demonstrates good robustness to messy text patterns."

        print(f"\n{conclusion}")
        print(f"\nThe final model achieves {self.best_accuracy*100:.2f}% accuracy, demonstrating")
        print(f"effective classification across all {len(class_names)} categories.")
        print(f"This provides a robust foundation for real-world text classification tasks.")

        print("\n" + "=" * 60)
        print(f"✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        return {
            'best_model': best_result['Model'],
            'best_features': best_result['Features'],
            'best_accuracy': self.best_accuracy,
            'impact_analysis': impact_analysis,
            'class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'class_names': class_names
            }
        }

    def run_complete_analysis(self):
        """Run the complete text classification analysis pipeline"""
        print("🚀 Starting Complete Text Classification Analysis...")
        print()

        start_time = time.time()

        # Setup
        setup_nltk()

        # Step 1: Load and explore data
        if not self.load_and_explore_data():
            return False

        # Step 2: Analyze messiness patterns
        self.analyze_messiness_patterns()

        # Step 3: Create preprocessing pipeline
        if not self.create_preprocessing_pipeline():
            return False

        # Step 4: Preprocess texts
        if not self.preprocess_texts():
            return False

        # Step 5: Engineer features
        if not self.engineer_features():
            return False

        # Step 6: Train and evaluate models
        results_df, best_result, y_test = self.train_and_evaluate_models()

        # Step 7: Analyze messiness impact
        impact_analysis = self.analyze_messiness_impact()

        # Step 8: Generate final report
        final_report = self.generate_final_report(results_df, best_result, y_test, impact_analysis)

        total_time = time.time() - start_time
        print(f"\n⏱️  Total Analysis Time: {total_time:.2f} seconds")

        return final_report

def main():
    """Main function to run the text classification analysis"""

    # Initialize pipeline
    pipeline = TextClassificationPipeline()

    # Run complete analysis
    try:
        results = pipeline.run_complete_analysis()

        if results:
            print("\n🎊 Text classification analysis completed successfully!")
            return results
        else:
            print("\n❌ Analysis failed. Please check the error messages above.")
            return None

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error during analysis: {e}")
        return None

if __name__ == "__main__":
    main()