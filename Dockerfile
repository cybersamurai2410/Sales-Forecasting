FROM python:3.11

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all project files into the container's working directory
COPY . /code

# Run the FastAPI application using fastapi
CMD ["fastapi", "run", "main.py", "--port", "80"]
