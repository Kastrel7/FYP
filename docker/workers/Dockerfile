FROM python:3.11-slim
WORKDIR /app
RUN pip install psutil
COPY worker.py .
CMD ["python3", "-u", "worker.py"]
