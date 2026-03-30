from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
CORS(app)

# Load model and real dataset at startup
model = joblib.load('model.pkl')
df = pd.read_csv('parkinsons.csv')

FEATURE_ORDER = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Random Forest Parkinson Classifier', 'dataset_rows': len(df)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400

        features = data['features']
        if len(features) != 22:
            return jsonify({'error': f'Expected 22 features, got {len(features)}'}), 400

        arr = np.array(features, dtype=float).reshape(1, -1)
        prediction = model.predict(arr)[0]
        probability = model.predict_proba(arr)[0]

        confidence = float(probability[int(prediction)]) * 100
        label = "Parkinson's Detected" if prediction == 1 else 'Healthy'

        return jsonify({
            'result': label,
            'prediction': int(prediction),
            'confidence': round(confidence, 1),
            'prob_healthy': round(float(probability[0]) * 100, 1),
            'prob_parkinsons': round(float(probability[1]) * 100, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-plots', methods=['POST'])
def generate_plots():
    try:
        X = df.drop(['name', 'status'], axis=1)[FEATURE_ORDER]
        y = df['status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                    xticklabels=['Healthy', "Parkinson's"],
                    yticklabels=['Healthy', "Parkinson's"],
                    linewidths=0.5, linecolor='#333')
        ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', color='white', pad=15)
        ax.set_xlabel('Predicted', color='#c084fc', fontsize=12)
        ax.set_ylabel('Actual', color='#c084fc', fontsize=12)
        ax.tick_params(colors='#e2e8f0')
        plt.setp(ax.get_xticklabels(), color='#e2e8f0')
        plt.setp(ax.get_yticklabels(), color='#e2e8f0')
        fig.tight_layout(pad=1.5)
        fig.savefig('confusion.png', dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)

        # --- Feature Importance (top 10) ---
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]
        feat_names = [FEATURE_ORDER[i] for i in sorted_idx][::-1]
        feat_vals  = importances[sorted_idx][::-1]

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')
        colors = plt.cm.magma(np.linspace(0.3, 0.85, len(feat_names)))
        bars = ax2.barh(feat_names, feat_vals, color=colors, height=0.6, edgecolor='none')
        for bar, val in zip(bars, feat_vals):
            ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', color='#e2e8f0', fontsize=9)
        ax2.set_title('Top 10 Feature Importances', fontsize=15, fontweight='bold', color='white', pad=15)
        ax2.set_xlabel('Importance Score', color='#c084fc', fontsize=12)
        ax2.tick_params(colors='#e2e8f0', axis='both')
        plt.setp(ax2.get_yticklabels(), color='#e2e8f0')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#333')
        ax2.spines['bottom'].set_color('#333')
        ax2.xaxis.grid(True, color='#2d2d4e', linestyle='--', alpha=0.6)
        ax2.set_axisbelow(True)
        fig2.tight_layout(pad=1.5)
        fig2.savefig('importance.png', dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig2)

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
