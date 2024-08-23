from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the tokenizer and model from the model directory
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')

# Set the model in evaluation mode
model.eval()

# Initialize an empty list to store sentiment history
sentiment_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    return inputs

def predict_sentiment(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits

def interpret_results(logits):
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()

    # Get the highest probability sentiment
    sentiment_idx = torch.argmax(probs).item()

    # Map the sentiment to a label
    labels = {0: "Neutral", 1: "Negative", 2: "Positive"}
    
    # Extract confidence for each sentiment
    confidences = probs.tolist()

    return labels[sentiment_idx], confidences


def generate_wordcloud(text_series):
    combined_text = " ".join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generate_sentiment_barchart(sentiment_counts):
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color=['red', 'gray', 'green'])
    plt.title('Sentiment Analysis Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def home():
    global sentiment_history
    
    if request.method == 'POST':
        text = request.form['text']
        logits = predict_sentiment(text)
        sentiment, confidences = interpret_results(logits)
        
        # Prepare the data structure with `text`, `sentiment`, and `confidences`
        sentiment_data = {
            'text': text,
            'sentiment': sentiment,
            'confidences': confidences
        }
        
        sentiment_history.insert(0, sentiment_data)
        
        if len(sentiment_history) > 5:
            sentiment_history = sentiment_history[:5]
        
        return render_template('index.html', sentiment=sentiment, confidences=confidences, text=text, history=sentiment_history)
    
    return render_template('index.html', history=sentiment_history)

@app.route('/analyze_csv', methods=['GET', 'POST'])
def analyze_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part")  # Debugging statement
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")  # Debugging statement
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to {filepath}")  # Debugging statement

            try:
                df = pd.read_csv(filepath)
                print(df.head())  # Debugging statement
            except pd.errors.ParserError as e:
                print(f"Error parsing CSV file: {e}")  # Debugging statement
                return render_template('analyze_csv.html', error=f'Error parsing CSV file: {e}')
            except Exception as e:
                print(f"An error occurred: {e}")  # Debugging statement
                return render_template('analyze_csv.html', error=f'An error occurred: {e}')

            if 'Text' not in df.columns:
                print("Missing 'Text' column")  # Debugging statement
                return render_template('analyze_csv.html', error="CSV file must contain a 'Text' column.")
            
            # Continue with analysis...
            results = []
            sentiments = []
            for sentence in df['Text']:
                sentiment, confidence = interpret_results(predict_sentiment(sentence))
                results.append({'sentence': sentence, 'sentiment': sentiment, 'confidence': confidence})
                sentiments.append(sentiment)

            # Generate the statistics and word cloud
            result_df = pd.DataFrame(results)
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            barchart_data = generate_sentiment_barchart(sentiment_counts)
            
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
            result_df.to_csv(result_filepath, index=False)

            return render_template('analyze_csv.html', result_file=f'result_{filename}', results=result_df.to_dict(orient='records'), barchart_data=barchart_data)
    
    return render_template('analyze_csv.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
