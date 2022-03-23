# example-mlflow
example of MLflow, a platform for the machine learning lifecycle

## Launch the environment

All three services (MySQL database,minio, MLflow and Nginx) can be runned by using docker-compose tool:

    docker-compose up -d

Then, got to:

    http://localhost:5001/#/
or 
    http://localhost/#/

## Use MLflow in Python

Create a venv and install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    python3 -m pip install poetry
    poetry install

Check carefully the example_experiment python script. It's straight easy to understand by reading the comments.

Now you have a beatiful UI to handle all of your experiments and a complete SQL base that contains everything to be usefull later ;)

## Note

- Do not delete the folder *mlflow_data* and *mysql_data* that have been created. It contains the MySQL docker data and the models registered by MLflow. It can be stored in a distant s3 folder, but that will cost money on GCP or Azure.
- An experiment name is UNIQUE. If you delete an experiment, you won't be able to use the same name again (or you have to delete in the SQL database everything related to it). Simply delete runs.
- Change values in the .env file ! It allows to connect to the MySQL and I did not secured the Nginx properly.
