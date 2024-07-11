import warnings
warnings.filterwarnings('ignore')

from flask import Flask
from flask import request, render_template, url_for, redirect,jsonify
import requests

from src.pipelines.prediction_pipeline import predictionPipeline

application = Flask(__name__)
app = application


@app.route("/",methods=["GET"])
def home():
    return redirect(url_for('predict'))

@app.route("/predict",methods=["GET","POST"])
def predict():
    if(request.method == "GET"):
        return render_template("predict.html")
    else:
        X = request.form.get("input_text")
        predict_pipeline = predictionPipeline()
        top_10_similar_items = predict_pipeline.predict(X=X)
        print(top_10_similar_items)
     
        # Return predictions and image URLs as JSON response
        return jsonify(top_10_similar_items)






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)