export MLFLOW_TRACKING_URI=http://localhost:5000

python3 src/model_trainer.py --phase-id phase-1 --prob-id prob-1 --task clf --type_model lgbm --time_tuning 500 --add-captured-data true