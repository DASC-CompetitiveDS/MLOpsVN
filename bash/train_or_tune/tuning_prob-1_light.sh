TIME_TUNING="${1:-1200}"
OPTFLAGNAME=""

if [ ! -z "$2" ]; then
   OPTFLAGNAME="--model_name $2"
fi

python3 src/model_trainer.py \
        --phase-id phase-3 \
        --prob-id prob-1 \
        --time_tuning $TIME_TUNING \
        --cross_validation True \
        --early_stopping_rounds 50 \
        --learning_rate 0.5 \
        $OPTFLAGNAME