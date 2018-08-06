FROM pccl/base:cpu
WORKDIR /opt/
COPY . .
RUN pip install -r ./requirements.txt
CMD python3 analogical_comparison.py 3
