FROM public.ecr.aws/docker/library/python:3.9
ARG PROJECT_ROOT

RUN apt-get update && apt-get install -y \
    graphviz

# Poetryの設定
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH /root/.local/bin:$PATH

RUN poetry config virtualenvs.in-project true