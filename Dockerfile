#Author: Paarth Sharma 
#Filename: Dockerfile 
#Project Name: Spotify Popularity Predictor
#Creation Date: Nov 23 2025
#Modification Date: Nov 25 2025
#Description: Creates a isolated self contained environment with all the dependencies allows running the project wihtout installing anything on the host kernel 
#base python image 
FROM python:3.10-slim

# Ensure system is updated
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#declaring the working directory 
WORKDIR /app

#copy the requirements 
COPY requirements.txt .

#install the requirements 
RUN pip install -r requirements.txt

#copy everything 
COPY . .

#expose the port 
EXPOSE 8501

#run the commands on the docker machine 
CMD bash -c "\
    echo 'Starting Spotify ML Pipeline...' && \
    python data_processing.py && \
    python train_model.py && \
    python evaluate_model.py && \
    echo 'Launching Streamlit App...' && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 \
"