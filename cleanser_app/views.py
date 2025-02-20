import pandas as pd
import numpy as np
import io
import chardet
import spacy
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.http import HttpResponse
from sklearn.ensemble import IsolationForest
from rapidfuzz import process
from io import BytesIO

# Load spaCy medical model
nlp = spacy.load("en_core_web_md")

# Medical dictionary for correction
medical_terms = {
    'cancer': 'cancer', 'cncer': 'cancer', 'paracetnol': 'paracetamol', 'paracetamol': 'paracetamol',
    'dabetes': 'diabetes', 'diabetes': 'diabetes', 'hypertenson': 'hypertension', 'hypertension': 'hypertension',
    'ashma': 'asthma', 'asthma': 'asthma', 'aspin': 'aspirin', 'aspirin': 'aspirin',
    'allergies': 'allergy', 'allergy': 'allergy', 'anemia': 'anaemia', 'anaemia': 'anaemia',
    'sugery': 'surgery', 'surgery': 'surgery', 'diarhea': 'diarrhea', 'diarrhea': 'diarrhea',
    'phneumonia': 'pneumonia', 'pneumonia': 'pneumonia'
}

# Function to correct medical terms
def correct_word(word):
    best_match = process.extractOne(word, medical_terms.keys(), score_cutoff=80)
    return medical_terms[best_match[0]] if best_match else word

def ai_correct_text(text):
    if isinstance(text, str):
        doc = nlp(text)
        corrected_tokens = [correct_word(token.text.lower()) for token in doc]
        return " ".join(corrected_tokens)
    return text

# Function to detect anomalies using Isolation Forest
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    numerical_cols = ['Age', 'BP', 'Billing Amount']

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Anomaly'] = model.fit_predict(df[numerical_cols])
    df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Anomalous' if x == -1 else 'Normal')
    return df

# Function to generate boxplots
def generate_boxplots(df):
    images = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Define color palette
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan']

    for i, column in enumerate(numeric_cols):
        plt.figure(figsize=(5, 4))  # Adjust figure size
        sns.boxplot(y=df[column], color=colors[i % len(colors)])  # Assign color from palette
        plt.title(f'Boxplot of {column}', fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save the plot to a BytesIO object
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        images.append(img_base64)

        plt.close()

    return images

# Function to generate Age Bar Graph, Blood Type Pie Chart, and BP Status Pie Chart
def generate_charts(df):
    charts = {}

    # Age Distribution
    if 'Age' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['Age'].dropna(), bins=10, kde=True)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        charts['age_bar'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

    # Blood Type Pie Chart
    if 'Blood Type' in df.columns:
        blood_counts = df['Blood Type'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(blood_counts, labels=blood_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Blood Type Distribution')

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        charts['blood_pie'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

    # BP Status Pie Chart
    if 'BP_Status' in df.columns:
        bp_counts = df['BP_Status'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(bp_counts, labels=bp_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('BP Status Distribution')

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        charts['bp_pie'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

    return charts

# Function to clean and format dates
def format_date(date_str):
    if not isinstance(date_str, str) or date_str.strip() == 'NA':
        return 'NA'
    try:
        parsed_date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        return parsed_date.strftime('%d/%m/%Y') if pd.notna(parsed_date) else 'Invalid Date'
    except Exception:
        return 'Invalid Date'

# Main data cleaning function


def clean_data(request):
    if request.method == 'POST' and request.FILES.get('dataset_file'):
        uploaded_file = request.FILES['dataset_file']
        file_name = uploaded_file.name.lower()
        file_data = uploaded_file.read()
        detected_encoding = chardet.detect(file_data)['encoding'] or 'utf-8'

        try:
            file_stream = io.BytesIO(file_data)
            df = pd.read_csv(file_stream, encoding=detected_encoding) if file_name.endswith('.csv') else pd.read_excel(file_stream)
        except Exception as e:
            return HttpResponse(f"Error reading file: {str(e)}")

        df.replace(["NA", "nan", "None", "N/A", "", "null"], pd.NA, inplace=True)

        # Generate Boxplots Before Cleaning
        boxplot_images = generate_boxplots(df)

        # AI-Based Medical Text Correction
        if 'Medical Condition' in df.columns:
            df['Medical Condition'] = df['Medical Condition'].apply(ai_correct_text)
        if 'Medication' in df.columns:
            df['Medication'] = df['Medication'].apply(ai_correct_text)

        # Date Formatting
        for column in df.columns:
            if df[column].dtype == 'object' and df[column].str.contains(r'\d{1,2}[-/\s][A-Za-z]*[-/\s]?\d{2,4}', na=False, regex=True).any():
                df[column] = df[column].apply(format_date)

        # Cleaning: Age, Blood Type, Gender, Name, BP
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce').apply(lambda x: 'NA' if pd.isna(x) or x < 1 or x > 130 else x)

        VALID_BLOOD_TYPES = {"A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"}
        if 'Blood Type' in df.columns:
            df['Blood Type'] = df['Blood Type'].apply(lambda x: x.upper().replace('POSITIVE', '+').replace('NEGATIVE', '-') if isinstance(x, str) else 'NULL')
            df['Blood Type'] = df['Blood Type'].apply(lambda x: x if x in VALID_BLOOD_TYPES else 'NULL')

        if 'Gender' in df.columns:
            gender_map = {'f': 'Female', 'female': 'Female', 'm': 'Male', 'male': 'Male', 'trans': 'Transgender'}
            df['Gender'] = df['Gender'].apply(lambda x: gender_map.get(x.lower(), 'NA') if isinstance(x, str) else 'NA')

        if 'Name' in df.columns:
            df['Name'] = df['Name'].apply(lambda x: " ".join([token.capitalize() for token in str(x).split()]) if isinstance(x, str) else "NA")

        if 'Billing Amount' in df.columns:
            df['Billing Amount'] = pd.to_numeric(df['Billing Amount'], errors='coerce').apply(lambda x: round(x, 2) if pd.notna(x) else 'NA')

        if 'BP' in df.columns:
            df['BP'] = pd.to_numeric(df['BP'], errors='coerce').fillna(120)
            df['BP_Status'] = df['BP'].apply(lambda bp: 'High BP' if bp > 120 else 'Normal')

        # Detect anomalies
        df = detect_anomalies(df)
        
        # Remove Duplicate Rows
        df.drop_duplicates(inplace=True)

        # Generate Charts
        charts = generate_charts(df)

        # Convert cleaned data to CSV
        cleaned_csv = io.StringIO()
        df.to_csv(cleaned_csv, index=False)
        cleaned_csv.seek(0)
        cleaned_csv_data = cleaned_csv.getvalue()
        cleaned_csv_base64 = base64.b64encode(cleaned_csv_data.encode()).decode()

        return render(request, 'index.html', {
            'cleaned_csv': f"data:text/csv;base64,{cleaned_csv_base64}",
            'boxplot_images': boxplot_images,
            'age_bar': charts.get('age_bar'),
            'blood_pie': charts.get('blood_pie'),
            'bp_pie': charts.get('bp_pie')
        })

    return render(request, 'index.html')

def index(request):
    return render(request, 'index.html')
