python3 src/model_trainer.py \
        --phase-id phase-3 \
        --prob-id prob-2 \
        --time_tuning 1000 \
        --log_confusion_matrix True \
        --cross_validation True \
        --learning_rate 0.2 \
        --early_stopping_rounds 50 \
        --model_name cv_lr-0.2