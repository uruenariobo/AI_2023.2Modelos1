# AI_2023.2Modelos1

# NYC Taxi Fare Prediction Challenge

Welcome to the NYC Taxi Fare Prediction Challenge repository! This project is part of the competition hosted on Kaggle, where the goal is to predict taxi fare amounts in New York City based on various features.

## Competition Overview

The [New York City Taxi Fare Prediction Challenge](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview) aims to improve the accuracy of fare amount predictions for taxi rides in NYC using machine learning techniques. The primary evaluation metric is the Root Mean-Squared Error (RMSE), where smaller values indicate better predictive performance.

The challenge is to develop predictive models that can estimate taxi fare amounts more accurately than basic distance-based estimates.

## Project Objectives

In this project, we will develop and deploy a predictive model to estimate taxi fare amounts. The first phase focuses on building the prediction model, and the Jupyter Notebook file for this phase is located in the `fase-1` directory. Below are the steps to execute Phase 1 successfully:

## Phase 1: Prediction Model

1. **Clone the Repository:**
   - Start by cloning this repository to your local environment using Git. Open your terminal and run the following command:

     ```bash
     git clone https://github.com/uruenariobo/AI_2023.2Modelos1.git
     ```

2. **Download Python 3.11:**
   - Make sure you have Python 3.11 installed on your system. You can download it from the official Python website [here](https://www.python.org/downloads/release/python-3110/).

3. **Set Up a Virtual Environment (Optional):**
   - To manage Python dependencies, consider creating a virtual environment. You can create one using the following commands:

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows, use "venv\Scripts\activate"
     ```

4. **Open the Project in Visual Studio Code:**
   - If you haven't already, download and install Visual Studio Code (VS Code) from the official website: [Visual Studio Code](https://code.visualstudio.com/).

   - Launch VS Code and open the project folder. You can do this by selecting "File" > "Open Folder" and selecting the folder where you cloned the repository.

5. **Execute Python Code:**
   - To run Python scripts in VS Code, open the Jupyter Notebook file `fase-1/nyc-taxi-fares-eda-modelling-2-93.ipynb`. You can execute code cells in the notebook individually by selecting the cell and clicking the "Run Cell" button (▶️) or using the keyboard shortcut (Shift+Enter).

   - Ensure that your Python environment is correctly configured, and the required packages are installed.

## Phase 2: Containerize Model

In this phase, we will configure a Docker container with all the necessary libraries to run the provided scripts. The container will include two main scripts:

1. `predict.py`: This script takes a CSV file as input and makes predictions using a pre-trained model stored on disk.

2. `train.py`: This script trains a new model using a dataset containing both input data and target values and saves the trained model.

### Prerequisites

Before you begin, make sure you have the following installed on your machine:

- [Docker](https://docs.docker.com/get-docker/)

### Creating the Docker Container

1. **Clone the project repository if you haven't already.**

   ```bash
   git clone https://github.com/uruenariobo/AI_2023.2Modelos1.git

2. **Navigate to the `phase-2` directory in your project using the following command:**

    ```bash
    cd AI_2023.2Modelos1/fase-2
    ```

3. **Build the Docker image using the provided Dockerfile with this command:**

    ```bash
    docker build -t taxi_model_container:latest .
    ```

This command will create a Docker image named `taxi_model_container:latest`.

You have successfully built a Docker image containing all the necessary components to run the Taxi Fare Prediction Model. In the next steps, you will learn how to make predictions and train new models using this Docker container.

4. **Running the Docker Container**

Now that you have built the Docker image containing the Taxi Fare Prediction Model, you can run the Docker container to make predictions or train new models.

### Running Predictions

To execute the `predict.py` script and make predictions, run the following command:

```bash
docker run -v $(pwd):/app taxi_model_container:latest python predict.py
```

This command mounts the current directory into the container, making it accessible to the script. You can provide your CSV data for predictions or use the default provided data.

### Training a New Model

If you have a dataset with input data and target values, you can train a new model using the train.py script. Here's how you can run the training script:

```bash
docker run -v $(pwd):/app taxi_model_container:latest python train.py
```

This command mounts the current directory into the container, making it accessible to the script. Make sure to provide your dataset as a CSV file named train.csv in the mounted directory.

### Model Storage

After training a new model using train.py, the trained model will be saved in the container. To retrieve the trained model from the container, use the following command:

```bash
docker cp <container_id>:/app/model.pkl /destination/path
```

Replace <container_id> with the actual container ID, and the model will be copied to the specified location on your local machine.

With this Docker container, you can easily make predictions and train new models using the provided scripts and libraries without worrying about library dependencies.

Note: Replace <container_id> and /destination/path with actual values when copying the model from the container.

**Conclusion:**

You have successfully set up the Taxi Fare Prediction Model in a Docker container. You can now run predictions and train new models within this container, making the process easy and efficient. Enjoy using the model for your projects!

To summarize, in this phase, you have:

- Built a Docker image with the necessary libraries and scripts.
- Run the Docker container for predictions and model training.
- Learned how to store the trained model from the container.

Feel free to use this model for your taxi fare prediction tasks.

## Phase 3: RESTful API

In Phase 3, we will deploy the Taxi Fare Prediction Model within a container and create a user-friendly RESTful API for making predictions. This API will provide a convenient and efficient way to access the model's predictive capabilities, allowing you to integrate it into various applications and services.

### Setting Up the RESTful API

1. **Clone the project repository if you haven't already.**

   ```bash
   git clone https://github.com/uruenariobo/AI_2023.2Modelos1.git

2. **Navigate to the `phase-3` directory in your project using the following command:**

    ```bash
    cd AI_2023.2Modelos1/fase-3
    ```

3. **Build the Docker image using the provided Dockerfile with this command:**

    ```bash
    docker build -t apirest:2 .
    ```

4. **Run the Docker container for the REST API:**

    ```bash
    docker run -p 5000:5000 apirest:2
    ```

### API Endpoints

The API exposes two endpoints:

1. **/predict: Returns predictions for a new data point.**

To make predictions, use the following curl command in your terminal:

```bash
    curl -X POST -H "Content-Type: application/json" -d '{"pickup_datetime": "2023-01-01 12:00:00", "pickup_longitude": -73.987, "pickup_latitude": 40.748, "dropoff_longitude": -74.001, "dropoff_latitude": 40.745, "passenger_count": 1}' http://localhost:5000/predict
```

Adjust the values as needed for your specific case.

2. **/train: Initiates a training process with standard training data.**

To train a new model with standard training data, use the following curl command:

```bash
    curl -X POST -H "Content-Type: application/json" -d '{"pickup_datetime": ["2023-01-01 12:00:00", "2023-01-01 12:15:00"], "pickup_longitude": [-73.987, -73.990], "pickup_latitude": [40.748, 40.750], "dropoff_longitude": [-74.001, -74.005], "dropoff_latitude": [40.745, 40.747], "passenger_count": [1, 2], "fare_amount": [10.0, 15.0]}' http://localhost:5000/train
```

Ensure that you adjust the values according to your actual training dataset.

With these steps, you have set up and run the Taxi Fare Prediction Model as a RESTful API within a Docker container. Enjoy using the API for making predictions and training new models seamlessly.


We're excited about the possibilities this API will offer to you. Get ready to experience the power of machine learning in a whole new way.

**Conclusion:**

Congratulations! You have successfully implemented the RESTful API for the Taxi Fare Prediction Model and can now interact with it using curl commands. The API offers a straightforward way to make predictions and train new models.
We hope you continue to enjoy being a part of this project and challenge. Happy modeling!


## Participants

- Miguel Angel Urueña Riobo
  - Email: miguel.uruena@udea.edu.co
  - Program: Ingeniería de Sistemas

## Dataset

You can access the competition dataset [here](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data).


