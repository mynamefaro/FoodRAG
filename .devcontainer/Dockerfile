FROM nvcr.io/nvidia/nemo:24.01.01.framework

# Create VSCode user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

#Allow user in tmp
RUN mkdir /tmp/user
RUN chmod 777 /tmp/user
RUN mkdir /tmp/.X11-unix
RUN chmod 777 /tmp/.X11-unix


# Switch to the VSCode user
USER $USERNAME

