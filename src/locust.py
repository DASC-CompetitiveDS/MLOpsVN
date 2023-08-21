from locust import HttpUser, task, between, constant_pacing
import json
from urllib.request import urlopen
import requests
import locust_plugins
#locust -f lotus_test.py -i 200
class PerformanceTests(HttpUser):
    wait_time = constant_pacing(1)
    @task(1)
    def test_model_pf(self):
        url = "https://raw.githubusercontent.com/MLOpsVN/mlops-mara-sample-public/main/data/curl/phase-1/prob-1/payload-1.json"
        response = requests.get(url)
        data = response.json()
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        self.client.post("/phase-1/prob-1/predict",
                               data=json.dumps(data),
                               headers=headers)
