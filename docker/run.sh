#!/bin/bash
docker run \
    -d \
    --init \
    -p8501:8501 \
    --rm \
    -it \
    --name=StreamlitTest \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$DATASET:/dataset \
    streamlit_test:latest \
    ${@-fish}
# docker run \
#     -d \
#     --init \
#     -p6006:6006 -p5000:5000 -p8888:8888 \
#     --rm \
#     -it \
#     --gpus=all \
#     --ipc=host \
#     --name=StreamlitTest \
#     --env-file=.env \
#     --volume=$PWD:/workspace \
#     --volume=$DATASET:/dataset \
#     streamlit_test:latest \
#     ${@-fish}
