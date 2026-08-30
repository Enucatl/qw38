FROM docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates=20260601~24.04.1 \
        cmake=3.28.3-1build7 \
        g++=4:13.2.0-7ubuntu1 \
        git=1:2.43.0-1ubuntu7.3 \
        ninja-build=1.11.1-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
