from flask import Flask, request, jsonify


app = Flask("sentiment")


def load_model():
    return 1

# Load the pretrained model
model = load_model()


@app.route('/predict', methods=['POST'])
def predict():
    # Sentence to predict sentiment for
    sentence = request.get_json()['sentence']

    response = {"sentiment": 1, "sentence": sentence, "confident": (0.7) * 100}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)