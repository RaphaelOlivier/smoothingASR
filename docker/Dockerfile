########## PyTorch 1 Deep Speech with extra dependencies for speech enhancement and post processing #################

ARG armory_version
FROM twosixarmory/pytorch-deepspeech:${armory_version} AS armory-pytorch-deepspeech-base

########## Extra packages for defense #############

WORKDIR /
RUN  git clone https://github.com/RaphaelOlivier/SCTK.git
WORKDIR /SCTK
RUN make config && make all && make check && make install && make doc
WORKDIR /tmp
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
WORKDIR /tmp/ctcdecode 
RUN pip install .
WORKDIR /tmp
RUN git clone https://github.com/RaphaelOlivier/pyaudlib.git --branch gard
WORKDIR /tmp/pyaudlib
RUN pip install . 
WORKDIR /tmp
RUN pip install tensorboardX
RUN git clone https://github.com/santi-pdp/ahoproc_tools.git
WORKDIR /tmp/ahoproc_tools
RUN pip install .
WORKDIR /tmp
RUN pip install --upgrade google-cloud-speech
RUN pip install pytorch_lightning 
RUN pip install git+https://github.com/romesco/hydra-lightning/#subdirectory=hydra-configs-pytorch-lightning
#WORKDIR /tmp
#RUN  git clone https://github.com/SeanNaren/warp-ctc.git
#RUN apt-get update && apt-get -y install cmake protobuf-compiler
#WORKDIR /tmp/warp-ctc
#ENV CUDA_HOME="/usr/local/cuda"
#RUN nvcc -V
#RUN  mkdir build
#WORKDIR /tmp/warp-ctc/build  
#RUN cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda .. && make
#WORKDIR /tmp/warp-ctc/pytorch_binding
#RUN python setup.py install

RUN git clone https://github.com/YiwenShaoStephen/espresso
WORKDIR /tmp/espresso
RUN git checkout v0.1-alpha
RUN pip install --editable .
RUN pip install kaldi_io
RUN pip install sentencepiece
RUN /opt/conda/bin/pip install adversarial-robustness-toolbox==1.7.2
RUN pip install kaldiio
WORKDIR /workspace