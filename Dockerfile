# TASK-001: pinned CUDA 13.4.x development environment.
#
# Base image: nvidia/cuda:13.4.1-devel-ubuntu24.04
# Manifest-list digest (immutable pin):
#   sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311
# linux/amd64 image digest:
#   sha256:f0eec85d23f7fdff7b6d74feb1d63e1e5921b0a92c3adfd9847875f830eb0382
#
# nvcc 13.4 accepts -std=c++23 only with GCC 14+ as the host compiler.
# Tool versions are recorded in docs/implementation/dev-environment.md.
# Do not float on a `latest` tag.

FROM nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja \
    CC=gcc-14 \
    CXX=g++-14 \
    CUDAHOSTCXX=g++-14

RUN printf '%s\n' \
        'Types: deb' \
        'URIs: https://snapshot.ubuntu.com/ubuntu/20260804T000000Z' \
        'Suites: noble noble-updates' \
        'Components: main universe restricted multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        '' \
        'Types: deb' \
        'URIs: https://snapshot.ubuntu.com/ubuntu/20260804T000000Z' \
        'Suites: noble-security' \
        'Components: main universe restricted multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
    > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake=3.28.3-1build7 \
        ninja-build=1.11.1-2 \
        gcc-14=14.2.0-4ubuntu2~24.04.1 \
        g++-14=14.2.0-4ubuntu2~24.04.1 \
        libstdc++-14-dev=14.2.0-4ubuntu2~24.04.1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
