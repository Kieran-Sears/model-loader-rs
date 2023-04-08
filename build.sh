#!/bin/bash

set -e

# Clone coqui-stt
git clone https://github.com/coqui-ai/STT.git --depth 1 --branch v1.0.2
cd STT

# Build the dependencies
cd deps/kenlm
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ../../../

# Build the main library
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# Install the library
sudo make install

# Update the linker cache
sudo ldconfig

# Test the installation
python3 ../bin/stt --model_path ../models/deepspeech-0.9.3-models.pbmm --scorer_path ../models/deepspeech-0.9.3-models.scorer --audio_path ../tests/data/audio/2830-3980-0043.wav
