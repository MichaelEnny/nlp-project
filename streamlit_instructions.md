# Streamlit Frontend for Text Classification

## Overview
This Streamlit application provides an interactive web interface for the text classification pipeline. Users can upload datasets, train models, and classify text in real-time.

## Features

### 📊 Dataset Management
- **Default Dataset**: Load the included dataset automatically
- **Custom Upload**: Upload your own CSV files for analysis
- **Data Validation**: Automatic format checking and error handling
- **Dataset Overview**: Interactive statistics and visualizations

### 🔍 Messiness Analysis
- **Pattern Detection**: Identify emojis, extra spaces, mixed case, etc.
- **Visual Analytics**: Interactive charts showing messiness patterns
- **Impact Assessment**: Understanding how messiness affects classification

### 🤖 Model Training
- **Multiple Algorithms**: Logistic Regression, Naive Bayes, Random Forest, SVM
- **Feature Engineering**: Bag of Words, TF-IDF, Character-level TF-IDF
- **Cross-Validation**: 5-fold stratified cross-validation
- **Progress Tracking**: Real-time training progress updates

### 🏆 Results Analysis
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Model Comparison**: Interactive charts comparing different approaches
- **Preprocessing Impact**: Before/after analysis of text cleaning
- **Best Model Selection**: Automatic identification of optimal configuration

### 🔮 Real-time Classification
- **Text Input**: Type or paste text for instant classification
- **Probability Scores**: See confidence levels for all categories
- **Preprocessing Preview**: View how text is cleaned before classification

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**:
   Open your browser to `http://localhost:8501`

## Usage Guide

### Step 1: Load Dataset
1. Choose "Use default dataset" or "Upload custom dataset" in the sidebar
2. Click the appropriate load button
3. Review the dataset overview in the first tab

### Step 2: Analyze Messiness
1. Navigate to the "Messiness Analysis" tab
2. Review the patterns detected in your text data
3. Understand the potential impact on classification

### Step 3: Train Models
1. Click "Train Models" in the sidebar
2. Wait for training to complete (progress bar shows status)
3. Review results in the "Results" tab

### Step 4: Classify New Text
1. Go to the "Classifier" tab
2. Enter text in the input box
3. Click "Classify Text" to get predictions

## Data Format Requirements

Your CSV file should have:
- **First column**: Text data (can contain messiness)
- **Second column**: Category labels
- **Headers**: Include column names

Example:
```csv
text,category
"OMG this game is AMAZING!!! 🏈⚽",sports
"New AI breakthrough in machine learning 🤖",tech
"Best pizza recipe ever!!! 🍕❤️",food
```

## Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Visualizations**: Plotly for interactive charts
- **Backend**: Integration with existing TextClassificationPipeline
- **State Management**: Streamlit session state for persistence

### Performance Optimizations
- **Caching**: Model results cached in session state
- **Lazy Loading**: Components loaded only when needed
- **Progress Feedback**: Real-time updates during training

### Error Handling
- **Input Validation**: Comprehensive checks for data format
- **Graceful Failures**: User-friendly error messages
- **Recovery Options**: Clear guidance for problem resolution

## Customization

### Styling
Modify the CSS in `app.py` to change:
- Color schemes
- Layout spacing
- Typography
- Component styling

### Functionality
Extend features by:
- Adding new model types
- Implementing additional preprocessing options
- Creating custom visualizations
- Adding export capabilities

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Dataset Loading Fails**:
   - Check CSV format matches requirements
   - Verify file encoding (UTF-8 recommended)

3. **Training Takes Too Long**:
   - Consider reducing dataset size for testing
   - Close other resource-intensive applications

4. **Classification Not Working**:
   - Ensure models are trained first
   - Check that input text is not empty after preprocessing

### Getting Help
- Check the Streamlit documentation: https://docs.streamlit.io/
- Review error messages in the browser console
- Ensure Python version compatibility (3.7+)

## Future Enhancements
- Model export/import functionality
- Batch text classification
- Advanced preprocessing options
- Performance benchmarking tools
- Integration with external APIs