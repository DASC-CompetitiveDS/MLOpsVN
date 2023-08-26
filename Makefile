SERVER = local

download_data_ci:
	bash bash/minio/get_data_ci.sh $(SERVER) phase-3 .

# teardown
teardown:
	make predictor_down
	make mlflow_down

monitoring_up:
	docker-compose -f deployment/monitoring/monitoring-compose.yaml up -d

monitoring_down:
	docker-compose -f deployment/monitoring/monitoring-compose.yaml down

# nginx
nginx_up:
	docker-compose -f deployment/nginx/docker-compose.yml up -d 

nginx_down:
	docker-compose -f deployment/nginx/docker-compose.yml down

# minio
# wait 2s to kill exited container
minio_up:
	docker-compose -f deployment/minio/docker-compose.yml up -d
	sleep 2
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
	bash deployment/deploy.sh model1 data/model_config/phase-3/prob-1/phase-3_prob-1_lgbm_cv_lr-0.5.yaml /phase-3/prob-1/predict 5001 data/predictor_config/phase-3/default_log.yaml $(SERVER)
	bash deployment/deploy.sh model2 data/model_config/phase-3/prob-2/phase-3_prob-2_lgbm_cv_lr-0.2.yaml /phase-3/prob-2/predict 5002 data/predictor_config/phase-3/default_log.yaml $(SERVER)

predictor_down:
	PORT=5001 docker-compose -f deployment/model_predictor/docker-compose.yml down
	PORT=5002 docker-compose -f deployment/model_predictor/docker-compose.yml down

predictor_restart:
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml stop
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml start

predictor_curl:
	curl -X POST http://localhost:5001/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:5002/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

predictor_curl_8000:
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
