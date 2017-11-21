import requests
import json

def test_sentiment():
	data = {"sentence":"i love koby so much and her"}
	response = requests.post('http://127.0.0.1:5000/predict', json=data)
	print(response.text)

if __name__ == '__main__':
	test_sentiment()
