
.PHONY: install train evaluate app clean

install:
	pip install -r requirements.txt

train:
	python train.py

evaluate:
	python evaluate.py

predict:
	python predict.py --input data/customer_support_tickets.csv --output predictions.csv

app:
	streamlit run app.py

clean:
	rm -rf __pycache__ .ipynb_checkpoints logs *.log
