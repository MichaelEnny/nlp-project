#!/usr/bin/env python3
"""
Quick run script for Text Classification Analysis

This script provides a simple way to run the text classification analysis
with different configurations and options.

Usage:
    python run_analysis.py                    # Run with default settings
    python run_analysis.py --help            # Show help
    python run_analysis.py --quick           # Quick analysis with fewer models
"""

import argparse
import sys
import os
from text_classification import TextClassificationPipeline

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Text Classification Analysis for Messy Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py                    # Full analysis
    python run_analysis.py --dataset custom.csv    # Custom dataset
    python run_analysis.py --quick           # Quick analysis
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/text classifcation.csv',
        help='Path to the dataset CSV file (default: dataset/text classifcation.csv)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis with fewer models'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save results (optional)'
    )

    return parser.parse_args()

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'sklearn', 'nltk', 'matplotlib', 'seaborn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTIP: Install them with: pip install -r requirements.txt")
        return False

    return True

def main():
    """Main function"""
    args = parse_arguments()

    print("Text Classification Analysis")
    print("=" * 50)

    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        print("Please ensure the dataset file exists.")
        return 1

    # Check requirements
    if not check_requirements():
        return 1

    print(f"Dataset: {args.dataset}")
    if args.quick:
        print("Quick mode: Running with reduced model set")

    try:
        # Initialize pipeline
        pipeline = TextClassificationPipeline(dataset_path=args.dataset)

        if args.quick:
            print("WARNING: Quick mode not yet implemented - running full analysis")

        # Run analysis
        results = pipeline.run_complete_analysis()

        if results:
            print("\nSUCCESS: Analysis completed successfully!")

            # Save results if output specified
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_results = {
                        'best_model': results['best_model'],
                        'best_features': results['best_features'],
                        'best_accuracy': float(results['best_accuracy']),
                        'impact_analysis': {
                            k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in results['impact_analysis'].items()
                        }
                    }
                    json.dump(serializable_results, f, indent=2)
                print(f"Results saved to: {args.output}")

            return 0
        else:
            print("\nERROR: Analysis failed!")
            return 1

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())