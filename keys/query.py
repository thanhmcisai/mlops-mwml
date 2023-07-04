from google.cloud import bigquery
from google.oauth2 import service_account

# Replace these with your own values
project_id = "mlops-course-391615"  # REPLACE
SERVICE_ACCOUNT_KEY_JSON = "keys/mlops-course-391615-d43fa6b1f66e.json"  # REPLACE

# Establish connection
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_JSON)
client = bigquery.Client(credentials=credentials, project=project_id)

# Query data
query_job = client.query(
    """
   SELECT *
   FROM mlops_course.labeled_projects"""
)
results = query_job.result()
print(results.to_dataframe().head())
