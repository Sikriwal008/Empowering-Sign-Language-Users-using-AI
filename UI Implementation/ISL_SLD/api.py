from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

@app.route('/correct_sentence', methods=['POST'])
def correct_sentence():
    data = request.json
    input_text = data.get("sentence", "")

    if not input_text.strip():
        return jsonify({"error": "Empty sentence"}), 400

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""The following string was formed using sequential hand gestures predicting characters in sign language. 
    Please correct the grammar and return the most likely intended sentence without explanation:
    
    \"{input_text}\"
    """

    try:
        response = model.generate_content(prompt)
        corrected = response.text.strip()
        print(corrected)
        return jsonify({"corrected_sentence": corrected})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
