# project id - replace with your GCP project id
PROJECT_ID=le-wagon-bootcamp-340910
# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-data-814-reddy
# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1
set_project:
	@gcloud config set project ${PROJECT_ID}
create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="/home/shilpa/code/getonGit/TaxiFareModel/raw_data/train_1k.csv"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
  # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
