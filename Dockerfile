FROM python:3.10-slim AS builder
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .


RUN pip install --no-cache-dir -v \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch


RUN pip install --no-cache-dir -v -r requirements.txt

# ---- Final stage: Lean runtime image ----
FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1

COPY app/ ./app/
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860"]