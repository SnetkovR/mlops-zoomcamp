FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pip install pipenv
RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

ENTRYPOINT [ "python", "starter.py", "2021", "4" ]