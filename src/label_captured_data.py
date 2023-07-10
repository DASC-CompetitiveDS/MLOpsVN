import os
import argparse
import mlflow
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from utils import AppConfig

def label_captured_data_cluster(prob_config: ProblemConfig, ratio_cluster: int):
    train_x = pd.read_parquet(prob_config.train_x_path)
    train_y = pd.read_parquet(prob_config.train_y_path)
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])

    captured_x = captured_x[train_x.columns.tolist()]
    np_captured_x = captured_x.to_numpy()
    n_captured = len(np_captured_x)
    n_samples = len(train_x) + n_captured
    logging.info(f"Loaded {n_captured} captured samples, {n_samples} train + captured")

    scaler_ = StandardScaler().fit(train_x)
    train_x = scaler_.transform(train_x)
    np_captured_x = scaler_.transform(np_captured_x)

    lbe = LabelEncoder().fit(train_y[prob_config.target_col])
    train_y_trans = lbe.transform(train_y[prob_config.target_col])

    logging.info("Initialize and fit the clustering model")
    n_cluster = int(len(train_x) / ratio_cluster) * len(np.unique(train_y))
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_cluster, random_state=prob_config.random_state
    ).fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(np_captured_x)

    logging.info(
        "Assign new labels to the new data based on the labels of the original data in each cluster"
    )
    new_labels = []
    for i in range(n_cluster):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y_trans[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # For a linear regression problem, use the mean of the labels as the new label
            # For a logistic regression problem, use the mode of the labels as the new label
            if ml_type == "regression":
                new_labels.append(np.mean(cluster_labels.flatten()))
            else:
                new_labels.append(
                    np.bincount(cluster_labels.flatten().astype(int)).argmax()
                )
    
    approx_label = lbe.inverse_transform([new_labels[c] for c in kmeans_clusters])
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])

    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)

    logging.info(f"Train set label ratio: {dict(train_y[prob_config.target_col].value_counts())}")
    logging.info(f"Test set label ratio (uncertained): {dict(approx_label_df[prob_config.target_col].value_counts())}")


def label_captured_data_model(prob_config: ProblemConfig, model_name: str):
    train_y = pd.read_parquet(prob_config.train_y_path)
    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
    if model_name == 'auto':
        list_experiment_ids = mlflow.search_experiments(filter_string=f"attribute.name LIKE '{prob_config.phase_id}_{prob_config.prob_id}%'")
        if len(list_experiment_ids) == 0:
            logging.info("There are no models available at this phase and this problem")
            return
        list_experiment_ids = [each.experiment_id for each in list_experiment_ids]
        client = mlflow.tracking.MlflowClient(tracking_uri=AppConfig.MLFLOW_TRACKING_URI)
        search = client.search_runs(experiment_ids=list_experiment_ids, order_by=["metrics.validation_score DESC"], max_results=1)[0]
        model_uri = f"runs:/{search.to_dictionary()['info']['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)
    else:
        try:
            model_uri = os.path.join("models:/", model_name)
            model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Loaded model {model_name}")
        except:
            logging.info(f"This model experiment doesn't exist {model_name}")
            return
        if  prob_config.prob_id not in model_name or prob_config.phase_id not in model_name:
            logging.info("The model phase or model problem doesn't match")
            return
        
    input_schema = mlflow.models.Model.load(model_uri).get_input_schema().to_dict()
    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])
    
    try:
        captured_x = captured_x[[each['name'] for each in input_schema]]
    except:
        logging.info("Some input features of the used model don't exist in captured data")
        return
    
    approx_label = model.predict(captured_x)
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])

    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)

    logging.info(f"Train set label ratio: {dict(train_y[prob_config.target_col].value_counts())}")
    logging.info(f"Test set label ratio (uncertained): {dict(approx_label_df[prob_config.target_col].value_counts())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--type_label", type=str, default="model", help="Biện pháp để label dữ liệu test, 2 phương pháp là ['cluster', 'model']")
    parser.add_argument("--ratio_cluster", type=int, default=20)
    parser.add_argument("--model_name", type=str, default='auto')
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    if args.type_label.lower() == 'cluster':
        label_captured_data_cluster(prob_config, args.ratio_cluster)
    elif args.type_label.lower() == 'model':
        label_captured_data_model(prob_config, args.model_name)
    else:
        logging.info("The available model type: [cluster, model]")
