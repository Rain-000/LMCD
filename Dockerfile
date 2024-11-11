FROM crpi-enao5v5vgst27ym7.cn-beijing.personal.cr.aliyuncs.com/ai_pipeline/miniconda3:latest

ADD ./environment.yml ./environment.yml

RUN conda install -n base -c conda-forge mamba && \
    mamba env update -n base -f ./environment.yml && \
    conda clean -afy


COPY ./test.py /app/test.py
COPY ./Data.py /app/Data.py
COPY ./LOSS.py /app/LOSS.py
COPY ./model.py /app/model.py
COPY ./test_Data.py /app/test_Data.py
COPY ./result.log /app/result.log
COPY ./Model1.pth /app/Model1.pth


WORKDIR /app


CMD ["python", "test.py"]
