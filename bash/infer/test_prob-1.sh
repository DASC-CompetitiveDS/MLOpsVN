python3 src/predictor.py \
        --config-path 'data/model_config/phase-3/prob-1/phase-3_prob-1_lgbm_cv_lr-0.5.yaml' \
        --path '/phase-3/prob-1/predict' \
        --port 8000 \
        --predictor-config-path 'data/predictor_config/phase-3/default_log.yaml' \
        --server local