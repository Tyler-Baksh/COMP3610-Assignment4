# COMP3610-Assignment4

## Prerequisites
1) Python 3.11+ installed
2) Docker Desktop installed and running
3) Trained baseline regression and tuned regression models from ([`COMP3610-Assignment2`](https://github.com/Tyler-Baksh/COMP3610-Assignment2)) downloaded, as well, the scaler and feature columns from the same assignment (in models/ folder)
4) A working Jupyter notebook environment

## Setup Instructions
1) Clone and run ([`COMP3610-Assignment2`](https://github.com/Tyler-Baksh/COMP3610-Assignment2)). .pkl files for the required models, scaler and feature columns will be generated in the `models/` folder
2) Clone this repository
3) Place the generated .pkl files from `COMP3610-Assignment2` in a `models` folder in this project

## Running the Project
1) Run the command `pip install -r requirements.txt`
2) Run the cells in the `Prerequisite` section
3) Run the command `mlflow ui --port 5000` and open http://localhost:5000 to view the MLflow dashboard
4) Run the cells in `Part 1: Model Tracking with MLflow`
5) For `Part 2: Model Serving with FastAPI` run the app using `uvicorn app:app --reload --port 8000`. Visit http://localhost:8000 to see the JSON response. Visit http://localhost:8000/docs to see the auto-generated Swagger UI
6) To run the docker container (ensuring you have Docker Desktop installed and running, the `models/` folder, and port 8001 free), run the command `docker compose up -d --build`
7) To stop the container, run the command `docker compose down`
