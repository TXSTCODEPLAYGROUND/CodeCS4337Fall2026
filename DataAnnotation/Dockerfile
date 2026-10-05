# Label Studio for the course, run locally (see README.md for the commands).
# The official image already contains Label Studio and its web server; this
# file only pins the version, so everyone gets the same interface, and turns
# on serving images from a folder of your computer.

# A fixed release, not "latest" (on Docker Hub, "latest" is a development build).
FROM heartexlabs/label-studio:1.23.2

# Let projects read images from /label-studio/files, where compose.yaml mounts
# the repository's data/annotation folder (Settings > Cloud Storage > Local files).
ENV LOCAL_FILES_SERVING_ENABLED=true \
    LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files

# The web interface listens on port 8080 inside the container.
EXPOSE 8080
