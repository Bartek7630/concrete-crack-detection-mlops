FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir --only-binary=:all: --default-timeout=100 -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY model_weights.pt ${LAMBDA_TASK_ROOT}
COPY class_mapping.json ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.handler" ]