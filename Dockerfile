# hash:sha256:cabe84b960ac5f3ddabca112f66a0f902643a3553a2892994b25ffc3940cc558
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils=2.0.9 \
        curl=7.68.0-1ubuntu2.13 \
        g++=4:9.3.0-1ubuntu2 \
        gcc=4:9.3.0-1ubuntu2 \
        python3-pip=20.0.2-5ubuntu1.6 \
        python3-setuptools=45.2.0-1 \
        python3-wheel=0.34.2-1 \
        wget=1.20.3-1ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        matplotlib==3.5.1 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    cython==0.29.32 \
    numpy==1.23.3 \
    pandas==1.5.0 \
    pycocotools==2.0.5 \
    scipy==1.9.1 \
    scikit-image \
    opencv-python

RUN /bin/bash -c "conda update -n base -c defaults conda && conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge"
CMD [ "/bin/bash" ]
