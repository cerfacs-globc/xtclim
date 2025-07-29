# docker run -it --mount src=`pwd`/'itwinai',target=/itwinai,type=bind xtclim
FROM buildpack-deps:bookworm

# Évite que les pipes masquent les erreurs
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Configuration des variables d'environnement
# gitleaks:allow
ARG GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D
ARG PYTHON_VERSION=3.10.14
ARG PYTHON_PIP_VERSION=23.0.1
ARG PYTHON_SETUPTOOLS_VERSION=65.5.1
ARG PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/dbf0c85f76fb6e1ab42aa672ffca6f0a675d9ee4/public/get-pip.py
ARG PYTHON_GET_PIP_SHA256=dfe9fd5c28dc98b5ac17979a953ea550cec37ae1b47a5116007395bfacff2ab9

ENV PATH=/usr/local/bin:$PATH \
    LANG=C.UTF-8

# Dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libbluetooth-dev \
        tk-dev \
        uuid-dev \
        git \
        wget \
        gnupg \
        dirmngr \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail temporaire pour la compilation de Python
WORKDIR /usr/src/python

RUN wget --progress=dot:giga -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" && \
    wget --progress=dot:giga -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" && \
    GNUPGHOME="$(mktemp -d)" && \
    gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY" && \
    gpg --batch --verify python.tar.xz.asc python.tar.xz && \
    gpgconf --kill all && \
    rm -rf "$GNUPGHOME" python.tar.xz.asc && \
    tar -xJf python.tar.xz --strip-components=1 && \
    rm python.tar.xz && \
    ./configure \
        --enable-loadable-sqlite-extensions \
        --enable-optimizations \
        --enable-option-checking=fatal \
        --enable-shared \
        --with-lto \
        --with-system-expat \
        --without-ensurepip && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

WORKDIR /
RUN rm -rf /usr/src/python

# Symlinks : python, idle, pydoc, etc.
RUN for bin in idle3 pydoc3 python3 python3-config; do \
      ln -svT "$bin" "/usr/local/bin/${bin%3}"; \
    done

# Installation de pip + setuptools depuis script vérifié
RUN wget --progress=dot:giga "$PYTHON_GET_PIP_URL" -O get-pip.py && \
    echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum -c - && \
    python get-pip.py \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-compile \
        "pip==$PYTHON_PIP_VERSION" \
        "setuptools==$PYTHON_SETUPTOOLS_VERSION" && \
    rm get-pip.py

# Copie et installation du projet Python
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Utilisateur non root
USER appuser
