# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
COPY duration_prediction/start.sh /usr/local/bin/start_service.sh
RUN chmod +x /usr/local/bin/start_service.sh
ENTRYPOINT ["start_service.sh"]
# CMD ["duration_prediction/start.sh"]