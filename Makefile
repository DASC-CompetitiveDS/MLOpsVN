SERVER = local

download_data_ci:
	bash bash/minio/get_data_ci.sh $(SERVER) phase-3 captured_data . test

# teardown
# teardown:
# 	make predictor_down
# 	make mlflow_down
# 	make nginx_down
# 	make minio_down
# 	make monitoring_down

down_all_platforms:
	make mlflow_down
	make nginx_down
	make minio_down
	make monitoring_down
	make airflow_up

up_all_platforms:
	make mlflow_up
	make nginx_up
	make minio_up
	make monitoring_up
	make airflow_down

monitoring_up:
	docker-compose -f deployment/monitoring/monitoring-compose.yaml up -d

monitoring_down:
	docker-compose -f deployment/monitoring/monitoring-compose.yaml down

airflow_up:
	sh deployment/airflow/deploy.sh

airflow_down:
	docker-compose -f deployment/airflow/docker-compose.yml down

# nginx
nginx_up:
	docker-compose -f deployment/nginx/docker-compose.yml up -d 

nginx_down:
	docker-compose -f deployment/nginx/docker-compose.yml down

# minio
# wait 2s to kill exited container
minio_up:
	docker-compose -f deployment/minio/docker-compose.yml up -d
	sleep 3
	docker-compose -f deployment/minio/docker-compose.yml rm -f

# minio_up:
# 	docker-compose -f deployment/minio/docker-compose.yml up -d

minio_down:
	docker-compose -f deployment/minio/docker-compose.yml up -d

# mlflow
mlflow_up:
	docker-compose -f deployment/mlflow/docker-compose.yml up -d

mlflow_down:
	docker-compose -f deployment/mlflow/docker-compose.yml down

# predictor
predictor_up:
	bash deployment/deploy.sh model1 data/model_config/phase-3/prob-1/phase-3_prob-1_lgbm_cv_specific_handle.yaml /phase-3/prob-1/predict 5001 data/predictor_config/phase-3/default.yaml $(SERVER)
	bash deployment/deploy.sh model2 data/model_config/phase-3/prob-2/phase-3_prob-2_lgbm_cv_lr-0.2.yaml /phase-3/prob-2/predict 5002 data/predictor_config/phase-3/default.yaml $(SERVER)

drift_monitoring_up:
	bash deployment/deploy_drift_monitoring.sh phase-3

predictor_curl:
	curl -X POST http://localhost:5040/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

# predictor_down:
# 	PORT=5001 docker-compose -f deployment/model_predictor/docker-compose.yml down
# 	PORT=5002 docker-compose -f deployment/model_predictor/docker-compose.yml down

# predictor_restart:
# 	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml stop
# 	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml start


predictor_curl_8000:
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

training_ci:
	sh bash/train_or_tune/tuning_prob-1_light.sh 60
	sh bash/train_or_tune/tuning_prob-2_light.sh 60