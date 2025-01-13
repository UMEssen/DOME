FROM python:3.9 as poetry2requirements
COPY pyproject.toml poetry.lock /
ENV POETRY_HOME=/etc/poetry
RUN pip3 install poetry
RUN pip3 install poetry-plugin-export
RUN python3 -m poetry export --without-hashes -f requirements.txt \
    | grep -v "torch=" \
    > /Requirements.txt


FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt-get update && \
    apt-get install -y locales && \
    sed -i -e 's/# de_DE.UTF-8 UTF-8/de_DE.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales

ENV LANG de_DE.UTF-8
ENV LC_ALL de_DE.UTF-8

# Install app dependencies
COPY --from=poetry2requirements /Requirements.txt /tmp
RUN pip3 install -U pip && \
    pip3 install -r /tmp/Requirements.txt && \
    rm /tmp/Requirements.txt

WORKDIR /app
ENV TRANSFORMERS_CACHE=/tmp/.cache/transformers
ENV HF_DATASETS_CACHE=/tmp/.cache/huggingface/datasets

COPY models /models
COPY config/TypeSystem.xml /app/config/TypeSystem.xml
COPY config/default-config.yaml /app/config/default-config.yaml

COPY dome /app/dome
