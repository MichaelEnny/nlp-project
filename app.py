#!/usr/bin/env python3
"""
Streamlit Frontend for Text Classification with Messy Data

This application provides an interactive web interface for the text classification
pipeline, allowing users to upload datasets, train models, and classify text.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import time
import pickle
from datetime import datetime
from text_classification import TextClassificationPipeline

# Page configuration
st.set_page_config(
    page_title="Text Classification Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False

def load_default_dataset():
    """Load the default dataset"""
    try:
        st.session_state.pipeline = TextClassificationPipeline()
        if st.session_state.pipeline.load_and_explore_data():
            st.session_state.dataset_loaded = True
            return True
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
    return False

def upload_dataset(uploaded_file):
    """Handle dataset upload"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize pipeline with uploaded dataset
        st.session_state.pipeline = TextClassificationPipeline(dataset_path=temp_path)
        if st.session_state.pipeline.load_and_explore_data():
            st.session_state.dataset_loaded = True
            st.success("Dataset uploaded and loaded successfully!")
            return True
    except Exception as e:
        st.error(f"Error uploading dataset: {e}")
    return False

def display_dataset_overview():
    """Display dataset overview and statistics"""
    if not st.session_state.dataset_loaded or not st.session_state.pipeline:
        return

    pipeline = st.session_state.pipeline
    df = pipeline.df

    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Samples</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df[pipeline.target_col].nunique()}</h3>
            <p>Categories</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        texts = df[pipeline.text_col].astype(str)
        avg_length = texts.str.len().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_length:.0f}</h3>
            <p>Avg Char Length</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_words = texts.str.split().str.len().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_words:.1f}</h3>
            <p>Avg Word Count</p>
        </div>
        """, unsafe_allow_html=True)

    # Target distribution
    st.markdown('<div class="sub-header">🎯 Category Distribution</div>', unsafe_allow_html=True)

    target_counts = df[pipeline.target_col].value_counts()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            title="Distribution of Categories",
            labels={'x': 'Category', 'y': 'Count'},
            color=target_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Category Counts:**")
        for category, count in target_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"• **{category}**: {count:,} ({percentage:.1f}%)")

        # Class balance
        balance_ratio = target_counts.min() / target_counts.max()
        if balance_ratio < 0.5:
            st.warning(f"⚠️ Class imbalance detected (ratio: {balance_ratio:.3f})")
        else:
            st.success(f"✅ Classes are balanced (ratio: {balance_ratio:.3f})")

def display_messiness_analysis():
    """Display messiness pattern analysis"""
    if not st.session_state.dataset_loaded or not st.session_state.pipeline:
        return

    st.markdown('<div class="sub-header">🔍 Text Messiness Analysis</div>', unsafe_allow_html=True)

    pipeline = st.session_state.pipeline
    texts = pipeline.df[pipeline.text_col].astype(str)

    # Define messiness patterns
    patterns = {
        'Emojis': texts.str.contains(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff]', regex=True),
        'Extra Spaces': texts.str.contains(r'\s{2,}', regex=True),
        'Mixed Case': texts.str.contains(r'[a-z]', regex=True) & texts.str.contains(r'[A-Z]', regex=True),
        'All Caps Words': texts.str.contains(r'\b[A-Z]{2,}\b', regex=True),
        'Numbers': texts.str.contains(r'\d', regex=True),
        'Special Chars': texts.str.contains(r'[^a-zA-Z0-9\s]', regex=True),
        'Repeated Chars': texts.str.contains(r'(.)\1{2,}', regex=True),
        'URLs': texts.str.contains(r'http[s]?://|www\.', regex=True, case=False),
        'Hashtags': texts.str.contains(r'#\w+', regex=True),
        'Mentions': texts.str.contains(r'@\w+', regex=True)
    }

    # Calculate pattern statistics
    pattern_stats = {}
    for pattern_name, pattern_mask in patterns.items():
        count = pattern_mask.sum()
        percentage = pattern_mask.mean() * 100
        pattern_stats[pattern_name] = {'count': count, 'percentage': percentage}

    # Create visualization
    col1, col2 = st.columns(2)

    with col1:
        pattern_names = list(pattern_stats.keys())
        pattern_percentages = [pattern_stats[name]['percentage'] for name in pattern_names]

        fig = px.bar(
            x=pattern_percentages,
            y=pattern_names,
            orientation='h',
            title="Messiness Patterns (%)",
            labels={'x': 'Percentage of Texts', 'y': 'Pattern Type'},
            color=pattern_percentages,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Pattern Details:**")
        for pattern_name, stats in pattern_stats.items():
            st.write(f"• **{pattern_name}**: {stats['count']:,} texts ({stats['percentage']:.1f}%)")

        # Overall messiness
        overall_messiness = sum(patterns.values()) > 0
        messy_count = overall_messiness.sum()
        messy_percentage = overall_messiness.mean() * 100

        st.info(f"**Overall Messiness**: {messy_count:,} texts ({messy_percentage:.1f}%) contain messiness patterns")

def run_training():
    """Run the complete training pipeline"""
    if not st.session_state.dataset_loaded or not st.session_state.pipeline:
        st.error("Please load a dataset first!")
        return

    pipeline = st.session_state.pipeline

    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Messiness analysis
        status_text.text("🔍 Analyzing messiness patterns...")
        progress_bar.progress(10)
        time.sleep(0.5)

        # Step 2: Preprocessing
        status_text.text("🛠️ Creating preprocessing pipeline...")
        progress_bar.progress(20)
        pipeline.create_preprocessing_pipeline()
        time.sleep(0.5)

        status_text.text("🔄 Preprocessing texts...")
        progress_bar.progress(40)
        pipeline.preprocess_texts()

        # Step 3: Feature engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(60)
        pipeline.engineer_features()

        # Step 4: Model training
        status_text.text("🤖 Training models...")
        progress_bar.progress(80)
        results_df, best_result, y_test = pipeline.train_and_evaluate_models()

        # Step 5: Analysis
        status_text.text("📊 Analyzing results...")
        progress_bar.progress(90)
        impact_analysis = pipeline.analyze_messiness_impact()

        # Step 6: Final report
        status_text.text("📋 Generating report...")
        progress_bar.progress(95)
        final_report = pipeline.generate_final_report(results_df, best_result, y_test, impact_analysis)

        progress_bar.progress(100)
        status_text.text("✅ Training completed successfully!")

        # Store results
        st.session_state.results = {
            'results_df': results_df,
            'best_result': best_result,
            'impact_analysis': impact_analysis,
            'final_report': final_report,
            'y_test': y_test
        }
        st.session_state.trained = True

        st.success("🎉 Model training completed successfully!")

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        progress_bar.empty()
        status_text.empty()

def display_results():
    """Display training results and visualizations"""
    if not st.session_state.trained or not st.session_state.results:
        return

    results = st.session_state.results
    results_df = results['results_df']
    best_result = results['best_result']
    impact_analysis = results['impact_analysis']

    st.markdown('<div class="sub-header">🏆 Best Model Results</div>', unsafe_allow_html=True)

    # Best model metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{best_result['Test_Accuracy']:.4f}</h3>
            <p>Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{best_result['CV_Mean']:.4f}</h3>
            <p>CV Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{impact_analysis['improvement_pct']:+.2f}%</h3>
            <p>Preprocessing Impact</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{best_result['Training_Time']:.1f}s</h3>
            <p>Training Time</p>
        </div>
        """, unsafe_allow_html=True)

    # Model comparison
    st.markdown('<div class="sub-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Performance comparison chart
        fig = px.bar(
            results_df,
            x='Test_Accuracy',
            y='Model',
            color='Features',
            title="Model Performance by Feature Type",
            labels={'Test_Accuracy': 'Test Accuracy', 'Model': 'Model Type'},
            orientation='h'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CV vs Test accuracy scatter
        fig = px.scatter(
            results_df,
            x='CV_Mean',
            y='Test_Accuracy',
            color='Features',
            size='Training_Time',
            hover_data=['Model'],
            title="Cross-Validation vs Test Accuracy",
            labels={'CV_Mean': 'CV Accuracy', 'Test_Accuracy': 'Test Accuracy'}
        )
        fig.add_shape(
            type="line",
            x0=results_df['CV_Mean'].min(),
            y0=results_df['CV_Mean'].min(),
            x1=results_df['CV_Mean'].max(),
            y1=results_df['CV_Mean'].max(),
            line=dict(dash="dash", color="gray")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Preprocessing impact analysis
    st.markdown('<div class="sub-header">🔧 Preprocessing Impact Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Before/after comparison
        comparison_data = {
            'Condition': ['Original (Messy)', 'Preprocessed (Clean)'],
            'Accuracy': [impact_analysis['messy_accuracy'], impact_analysis['clean_accuracy']]
        }

        fig = px.bar(
            comparison_data,
            x='Condition',
            y='Accuracy',
            title="Preprocessing Impact on Accuracy",
            color='Condition',
            color_discrete_map={
                'Original (Messy)': '#e74c3c',
                'Preprocessed (Clean)': '#27ae60'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Impact Summary:**")
        st.write(f"• **Original accuracy**: {impact_analysis['messy_accuracy']:.4f}")
        st.write(f"• **Clean accuracy**: {impact_analysis['clean_accuracy']:.4f}")
        st.write(f"• **Improvement**: {impact_analysis['improvement']:+.4f} ({impact_analysis['improvement_pct']:+.2f}%)")
        st.write(f"• **Impact level**: {impact_analysis['impact_level']}")
        st.write(f"• **Recommendation**: Preprocessing is {impact_analysis['recommendation']}")
        st.write(f"• **Vocabulary reduction**: {impact_analysis['vocab_reduction']:.1f}%")

def text_classifier_interface():
    """Interactive text classification interface"""
    if not st.session_state.trained or not st.session_state.pipeline:
        st.warning("Please train a model first to use the text classifier!")
        return

    st.markdown('<div class="sub-header">🔮 Real-time Text Classification</div>', unsafe_allow_html=True)

    pipeline = st.session_state.pipeline

    # Text input
    user_text = st.text_area(
        "Enter text to classify:",
        placeholder="Type or paste your text here...",
        height=100
    )

    if user_text and st.button("Classify Text", type="primary"):
        try:
            # Preprocess the text
            cleaned_text = pipeline.preprocessor.clean_text(user_text)

            if not cleaned_text.strip():
                st.error("Text becomes empty after preprocessing. Please try different text.")
                return

            # Get best model and vectorizer
            best_features = st.session_state.results['best_result']['Features']

            # Map feature names to vectorizers
            feature_map = {
                'Bag of Words': 'bow',
                'TF-IDF': 'tfidf',
                'Character TF-IDF': 'char_tfidf'
            }

            vectorizer_key = feature_map[best_features]
            vectorizer = pipeline.vectorizers[vectorizer_key]

            # Transform text
            text_features = vectorizer.transform([cleaned_text])

            # Predict
            prediction = pipeline.best_model.predict(text_features)[0]
            probabilities = pipeline.best_model.predict_proba(text_features)[0]

            # Get class name
            predicted_class = pipeline.label_encoder.classes_[prediction]

            # Display results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.success(f"**Predicted Category**: {predicted_class}")
                st.info(f"**Confidence**: {probabilities[prediction]:.4f}")

                st.markdown("**Preprocessing Result:**")
                st.text_area("Cleaned text:", cleaned_text, height=80, disabled=True)

            with col2:
                # Probability distribution
                prob_data = {
                    'Category': pipeline.label_encoder.classes_,
                    'Probability': probabilities
                }

                fig = px.bar(
                    prob_data,
                    x='Probability',
                    y='Category',
                    orientation='h',
                    title="Classification Probabilities",
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Classification failed: {e}")

def main():
    """Main Streamlit application"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">📊 Text Classification Analysis</div>', unsafe_allow_html=True)
    st.markdown("**Interactive analysis of text classification with messy data**")

    # Sidebar
    st.sidebar.markdown("## 🛠️ Configuration")

    # Dataset selection
    dataset_option = st.sidebar.radio(
        "Choose dataset option:",
        ["Use default dataset", "Upload custom dataset"]
    )

    if dataset_option == "Use default dataset":
        if st.sidebar.button("Load Default Dataset", type="primary"):
            with st.spinner("Loading default dataset..."):
                load_default_dataset()

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with text data and categories"
        )

        if uploaded_file is not None:
            if st.sidebar.button("Load Uploaded Dataset", type="primary"):
                with st.spinner("Processing uploaded dataset..."):
                    upload_dataset(uploaded_file)

    # Training section
    st.sidebar.markdown("## 🤖 Model Training")

    if st.session_state.dataset_loaded:
        if not st.session_state.trained:
            if st.sidebar.button("Train Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    run_training()
        else:
            st.sidebar.success("✅ Models trained successfully!")

            if st.sidebar.button("Retrain Models"):
                st.session_state.trained = False
                st.session_state.results = None
                st.experimental_rerun()

    # Main content tabs
    if st.session_state.dataset_loaded:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 Messiness Analysis", "🏆 Results", "🔮 Classifier"])

        with tab1:
            display_dataset_overview()

        with tab2:
            display_messiness_analysis()

        with tab3:
            if st.session_state.trained:
                display_results()
            else:
                st.info("Please train the models first to see results.")

        with tab4:
            text_classifier_interface()

    else:
        st.info("👆 Please load a dataset using the sidebar to get started.")

        # Show sample data format
        st.markdown("### 📋 Expected Data Format")
        st.markdown("Your CSV file should have the following structure:")

        sample_data = pd.DataFrame({
            'text': [
                'This is a sample sports text about football!!! 🏈',
                'Breaking: new tech startup raises $100M funding 💰',
                'Recipe: how to make amazing pasta 🍝'
            ],
            'category': ['sports', 'tech', 'food']
        })

        st.dataframe(sample_data, use_container_width=True)

        st.markdown("""
        **Requirements:**
        - First column: Text data (can be messy with emojis, extra spaces, etc.)
        - Second column: Category labels
        - CSV format with headers
        """)

if __name__ == "__main__":
    main()