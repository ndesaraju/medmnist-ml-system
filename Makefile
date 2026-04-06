#---------------------------------------------------
# Run the model pipeline
#---------------------------------------------------

.PHONY: setup data preprocess train test serve all clean

# Set up environment (run once)
setup:
	python3 -m venv venv
	. venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt

# Download the data
data:
	python -m src.data.make_dataset

# Preprocess the data
preprocess:
	python -m src.preprocess.build_features

# Train the model
train:
	python -m src.model.train

# Make predictions on the test data
test:
	python -m src.model.predict
# Serve the model
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Run full pipeline
all: data preprocess train test

#---------------------------------------------------
# Cleaning
#---------------------------------------------------

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf models/*
	rm -rf results/*