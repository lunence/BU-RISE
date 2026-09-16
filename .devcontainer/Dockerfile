# .devcontainer/Dockerfile

FROM ros:humble

# Install pip
RUN apt update && apt install -y python3-pip

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Added ros-humble-rmw-cyclonedds-cpp to match the LIMO's DDS implementation
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-tf-transformations \
    python3-transforms3d \
    curl \
    unzip \
    wget \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Add the Zenoh repository and install 1.8.0
RUN apt-get update && apt-get install -y gnupg curl && \
    mkdir -p /etc/apt/keyrings && \
    curl -L https://download.eclipse.org/zenoh/debian-repo/zenoh-public-key | gpg --dearmor --yes --output /etc/apt/keyrings/zenoh-public-key.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/zenoh-public-key.gpg] https://download.eclipse.org/zenoh/debian-repo/ /" | tee /etc/apt/sources.list.d/zenoh.list > /dev/null && \
    apt-get update && \
    apt-get install -y zenoh-bridge-ros2dds=1.8.0 && \
    rm -rf /var/lib/apt/lists/*

# Convenience script to connect to the Mac host's zenoh bridge
# host.docker.internal routes from the container directly to the macOS localhost
RUN echo '#!/bin/bash\nzenoh-bridge-ros2dds -e tcp/host.docker.internal:7447 "$@"' > /usr/local/bin/connect-zenoh && \
    chmod +x /usr/local/bin/connect-zenoh

# Source ROS 2 in new bash sessions
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc