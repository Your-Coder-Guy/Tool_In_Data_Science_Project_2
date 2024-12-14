# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "seaborn",
#   "python-dotenv",
#   "requests",
# ]
# ///

#!/usr/bin/env python3
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.environ.get("AIPROXY_TOKEN")

def load_csv(file_path):
    try:
        # Try default UTF-8 encoding first
        df = pd.read_csv(file_path)
        print("CSV file loaded successfully with UTF-8 encoding.")
        return df
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Trying 'latin1' encoding...")
        try:
            # Fallback to 'latin1' encoding
            df = pd.read_csv(file_path, encoding='latin1')
            print("CSV file loaded successfully with 'latin1' encoding.")
            return df
        except Exception as e:
            print(f"Error loading file with 'latin1' encoding: {e}")
            exit()
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def classify_columns(df):
    id_cols = [col for col in df.columns if "id" in col.lower() or "code" in col.lower()]
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns.difference(id_cols)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime', 'object']).apply(lambda col: pd.to_datetime(col, errors='coerce')).dropna(axis=1).columns
    return id_cols, numerical_cols, categorical_cols, date_cols

def analyze_data(df):
    
    id_cols, numerical_cols, categorical_cols, date_cols = classify_columns(df)

    insights = {
        "overview": {
            "num_rows": df.shape[0],
            "num_cols": df.shape[1],
            "missing_values": df.isnull().sum().to_dict()
        },
        "numerical_summary": df[numerical_cols].describe().to_dict(),
        "categorical_summary": {col: df[col].value_counts().to_dict() for col in categorical_cols}
    }

    return insights

def summarize_insights(insights):
    
    summarized_insights = {
        "overview": insights["overview"],
        "numerical_summary": {key: insights["numerical_summary"][key] for key in list(insights["numerical_summary"])[:3]},  # Limit to 3 columns
        "categorical_summary": {key: list(insights["categorical_summary"][key].keys())[:5] for key in list(insights["categorical_summary"])[:2]}  # Limit to 2 categories, 5 values each
    }
    return summarized_insights

def generate_story(insights, csv_name):
    
    summarized_insights = summarize_insights(insights)

    prompt = (
        f"You are a data storyteller. Analyze the following summarized insights from a dataset named '{csv_name}':\n\n"
        f"Overview: {summarized_insights['overview']}\n\n"
        f"Numerical Summary: {summarized_insights['numerical_summary']}\n\n"
        f"Categorical Summary: {summarized_insights['categorical_summary']}\n\n"
        f"Write a brief and engaging story, no longer than 700 words, summarizing the dataset and its potential meaning."
    )

    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        response = requests.post(r"http://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, 
                                 json={"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a creative storyteller."},
                                                                             {"role": "user", "content": prompt}]})
        response = response.json()
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating story: {e}")
        return "Story generation failed."

def save_story_and_images(csv_path, insights, story):
    
    # Create folder based on CSV file name
    csv_name = Path(csv_path).stem
    folder_path = Path(csv_name)
    folder_path.mkdir(exist_ok=True)

    # Save README.md with story
    readme_path = folder_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# Story for {csv_name}\n\n")
        f.write(story)

    # Generate and save visualizations
    numerical_cols = pd.DataFrame(insights['numerical_summary'])
    for col in numerical_cols.columns:
        plt.figure()
        sns.histplot(numerical_cols[col], kde=True)
        img_path = folder_path / f"{col}_distribution.png"
        plt.savefig(img_path)
        plt.close()

        # Add image reference to README
        with open(readme_path, "a") as f:
            f.write(f"![{col} Distribution](./{col}_distribution.png)\n")

    print(f"Story and images saved in folder: {folder_path}")

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Autolysis: Automatic Data Analysis and Storytelling")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to analyze")
    args = parser.parse_args()

    # Load CSV and process
    csv_path = args.csv_path
    df = load_csv(csv_path)
    insights = analyze_data(df)
    csv_name = Path(csv_path).stem
    story = generate_story(insights, csv_name)
    save_story_and_images(csv_path, insights, story)

if __name__ == "__main__":
    main()
