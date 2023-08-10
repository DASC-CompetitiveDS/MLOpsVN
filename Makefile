# teardown
teardown:
	make predictor_down
	make mlflow_down

# nginx
nginx_up:
	docker-compose -f deployment/nginx/docker-compose.yml up -d

nginx_down:
	docker-compose -f deployment/nginx/docker-compose.yml down

# mlflow
mlflow_up:
	docker-compose -f deployment/mlflow/docker-compose.yml up -d

mlflow_down:
	docker-compose -f deployment/mlflow/docker-compose.yml down

# predictor
predictor_up:
	bash deployment/deploy.sh model1 data/model_config/phase-3/prob-1/phase-3_prob-1_lgbm_cv_lr-0.5.yaml /phase-3/prob-1/predict 5001 False 
	bash deployment/deploy.sh model2 data/model_config/phase-3/prob-2/phase-3_prob-2_lgbm_cv_lr-0.2.yaml /phase-3/prob-2/predict 5002 False

predictor_down:
	docker-compose -f deployment/model_predictor/docker-compose.yml down

predictor_restart:
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml stop
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml start

predictor_curl:
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

predictor_curl_ip:
	curl -X POST http://20.205.210.58:5040/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://20.205.210.58:5040/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

predictor_curl_8000:
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json


predictor_curl_20:
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json & 
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json &
	curl -X POST http://localhost:5041/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json

predictor_curl_8000_20:
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-1/payload-1.json
	curl -X POST http://localhost:8000/phase-3/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-3/prob-2/payload-1.json
