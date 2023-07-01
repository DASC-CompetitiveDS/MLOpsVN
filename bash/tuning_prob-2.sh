export MLFLOW_TRACKING_URI=http://localhost:5000

python3 src/model_trainer.py \
        --phase-id phase-1 \
        --prob-id prob-2 \
        --time_tuning 100