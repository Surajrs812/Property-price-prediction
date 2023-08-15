# Flask app integrating front and back end in a local server

import json
from flask import Flask, render_template,request
from main import predict_price


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'GET':
        # Load location options from columns.json
        with open('/home/suraj/Programs/property_price/venv/static/columns.json') as json_file:
            data = json.load(json_file)
            locations = data['locations']
        
        return render_template('app.html', predicted_price='', locations=locations)
    else:
        location = str(request.form['location'])
        square_feet = float(request.form['square_feet'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        price = predict_price(location, square_feet, bathrooms, bedrooms)
        return render_template('app.html', predicted_price=f'Predicted Price in Lakhs: {price :.2f}')


if __name__ == "__main__":
    app.run(debug=True)
    