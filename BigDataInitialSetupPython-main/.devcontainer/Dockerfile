# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Locale setup
RUN apt-get update && apt-get install -y locales \
 && locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create non-root user `dev`
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install system packages and Python
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    build-essential \
    graphviz \
    curl \
    bash-completion \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
 && groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME \
 && apt-get clean

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir pdoc

# Use non-root user
USER $USERNAME
WORKDIR /workspaces/BigDataInitialSetupPython
