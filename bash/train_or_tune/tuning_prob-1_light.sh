TIME_TUNING="${1:-1200}"

python3 src/model_trainer.py \
        --phase-id phase-3 \
        --prob-id prob-1 \
        --time_tuning $TIME_TUNING \
        --cross_validation True \
        --early_stopping_rounds 50 \
        --learning_rate 0.5 \
        --model_name test_ci
