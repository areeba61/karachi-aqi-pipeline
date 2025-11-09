# app.py
from flask import Flask, render_template_string, request
from model_loader import forecast_df, full_df, daily_avg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <body style="background-color:#d3d3d3;">
    <h1 style="text-align:center;"> Karachi AQI Forecast</h1>
    <form action="/predict" method="post" style="text-align:center;">
        <button type="submit">Show AQI Forecast</button>
    </form>
    </body>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Forecast table
        forecast_table = forecast_df[["date", "hour", "aqi"]].to_html(index=False)
        forecast_html = f'<div style="display:flex; justify-content:center;">{forecast_table}</div>'

        # Daily average table with category
        avg_table = daily_avg.to_html(index=False)
        avg_html = f'<div style="display:flex; justify-content:center;">{avg_table}</div>'

        # AQI value chart
        fig, ax = plt.subplots(figsize=(12, 5))
        for label, group in full_df.groupby("source"):
            ax.plot(group["timestamp"], group["aqi"], label=label)
        ax.set_title("AQI Values: Historical + Forecast")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("AQI")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Render everything
        html = f"""
        <body style="background-color:#d3d3d3;">
        <h2 style="text-align:center;"> Hourly AQI Forecast Table</h2>
        {forecast_html}
        <h2 style="text-align:center;"> Daily AQI Averages + Category</h2>
        {avg_html}
        <h2 style="text-align:center;"> Full AQI Value Chart</h2>
        <div style="text-align:center;">
            <img src="data:image/png;base64,{plot_url}" />
        </div>
        </body>
        """
        return render_template_string(html)

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
