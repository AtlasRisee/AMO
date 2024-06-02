# shellcheck disable=SC2164
cd /app
python data_creation.py
python model_preprocessing.py
python model_preparation.py
python model_testing.py