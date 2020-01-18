FROM tensorflow/tensorflow:latest-py3

# update pip
RUN pip install --upgrade pip

# install  xlrd für Excel
RUN mkdir ./xlrd && \
    cd ./xlrd && \
    pip install xlrd

# install matplotlib
RUN mkdir ./matplotlib && \
    cd ./matplotlib && \
    pip install matplotlib

# install panda
RUN mkdir ./pandas && \
    cd ./pandas && \
    pip install pandas

# install ipython
RUN mkdir ./ipython && \
    cd ./ipython && \
    pip install ipython

# install sklearn
RUN mkdir ./sklearn && \
    cd ./sklearn && \
    pip install sklearn 

# install openpyxl
RUN mkdir ./openpyxl && \
    cd ./openpyxl && \
    pip install openpyxl

# install tensorflow-probability
RUN mkdir ./tensorflow-probability && \
    cd ./tensorflow-probability && \
    pip install tensorflow-probability

ADD . /developer

LABEL maintainer="Inijanna@gmail.com"