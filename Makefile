# teardown
teardown:
	make predictor_down
	make mlflow_down

# mlflow
mlflow_up:
	docker-compose -f deployment/mlflow/docker-compose.yml up -d

mlflow_down:
	docker-compose -f deployment/mlflow/docker-compose.yml down

# predictor
predictor_up:
	bash deployment/deploy.sh run_predictor data/model_config/phase-2/prob-1/phase-2_prob-1_lgbm__.yaml data/model_config/phase-2/prob-2/phase-2_prob-2_lgbm__.yaml 5041

predictor_down:
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml down

predictor_restart:
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml stop
	PORT=5041 docker-compose -f deployment/model_predictor/docker-compose.yml start

predictor_curl:
	curl -X POST http://localhost:5041/phase-2/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-2/prob-1/payload-1.json
	curl -X POST http://localhost:5041/phase-2/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-2/prob-1/payload-2.json
	curl -X POST http://localhost:5041/phase-2/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-2/prob-2/payload-1.json