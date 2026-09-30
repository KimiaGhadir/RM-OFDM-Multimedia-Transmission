# RM-OFDM Multimedia Transmission

This repository contains the implementation of image and video transmission using Random Modulation (RM) and OFDM in GNU Radio.

It includes complete pipelines for image and video transmission, packet generation, channel simulation, signal recovery, reconstruction, performance evaluation, and graphical user interfaces for easier testing.

---

## Features

### Image Transmission

- Image resizing and preprocessing
- Automatic BIN file generation
- Packetization and whitening
- RM transmission
- OFDM transmission
- Combined RM/OFDM mode
- OAMP-based equalization
- Image reconstruction
- BER calculation
- PSNR calculation
- PyQt5 graphical user interface

### Video Transmission

- Video frame extraction
- JPEG-based frame compression
- Video packetization
- Frame ID and packet ID handling
- RM transmission
- OFDM transmission
- OAMP equalization
- Packet validation and reordering
- Frame reconstruction
- Final video reconstruction
- BER calculation
- Frame recovery statistics
- PyQt5 graphical user interface

---

## Project Structure

RM-OFDM-Multimedia-Transmission/
|
|-- image/
|   |-- image_radio_ui.py
|   |-- rx_ran_ofdm.py
|   `-- ...
|
|-- video/
|   |-- ui.py
|   |-- Ran_ofdm.py
|   |-- prepare_and_run.sh
|   |-- prepare_video_packets.py
|   |-- run_video_until_complete.py
|   `-- ...
|
|-- custom_module/
|   |-- ber_cal.py
|   |-- data_separation.py
|   |-- OAMP_Equalizer.py
|   |-- Random_Modulation.py
|   |-- video_receiver.py
|   `-- ...
|
`-- README.md

---

## Image Transmission

The image transmission pipeline supports RM, OFDM, and combined RM/OFDM transmission.

The interface allows the user to:
- Select an input image
- Set image width and height
- Set packet length
- Generate BIN files automatically
- Select RM, OFDM, or COMB mode
- Run the GNU Radio simulation
- View the reconstructed images
- Compare BER and PSNR results

Typical test configuration:

Image Size: 128 x 128
Packet Length: 274
Mode: COMB

The BIN generation is integrated directly into the UI, which makes the full image transmission workflow easier to run.

---

## Video Transmission

For video transmission, each input video frame is compressed and divided into packets.

Each packet contains frame-level and packet-level information so that received data can be correctly reconstructed at the receiver.

The receiver handles:
- Frame identification
- Packet identification
- Packet ordering
- Missing packet handling
- Frame reconstruction
- JPEG decoding
- Final video generation

RM and OFDM branches are processed separately and their results are evaluated independently.

Example result from a noise-free 229-frame video test:

RM:
Decoded: 229/229
Exact:   229
BER:     0

OFDM:
Decoded: 229/229
Exact:   229
BER:     0

---

## Main Components

- GNU Radio
- Python
- NumPy
- OpenCV
- PyQt5
- Pillow
- Custom GNU Radio modules
- OAMP Equalizer
- Random Modulation

---

## Requirements

The project was developed and tested on Linux.

Main software requirements:

Python 3
GNU Radio
NumPy
OpenCV
PyQt5
Pillow

Depending on the GNU Radio installation, the custom modules used in this project may need to be installed in the GNU Radio Python module path.

---

## Running the Image Interface

Go to the image project folder:

cd image

Run:

python3 image_radio_ui.py

Then:
1. Select the input image.
2. Set the image width and height.
3. Set the packet length.
4. Click Generate BIN Files.
5. Select RM, OFDM, or COMB.
6. Select the GNU Radio Python file if needed.
7. Click Start Transmission.

The reconstructed output images and performance results will be displayed in the interface.

---

## Running the Video Interface

Go to the video project folder:

cd video

Run:

python3 ui.py

Then:
1. Select the input video.
2. Set the required simulation parameters.
3. Set the noise level.
4. Start the simulation.
5. Wait for RM and OFDM reconstruction to complete.

The reconstructed videos and transmission statistics will be displayed after processing.

---

## Packet Handling

### Image

The image pipeline converts the input image into packets and generates whitened BIN files for transmission.

For combined transmission, RM and OFDM branches are packed into the same packet structure.

### Video

The video pipeline stores information such as:
- Frame ID
- Packet ID
- Total packet count
- Frame size

This information is used to reconstruct frames in the correct order at the receiver.

---

## Performance Metrics

### Image
- BER
- PSNR
- Reconstructed image quality

### Video
- BER
- Number of decoded frames
- Number of exact reconstructed frames
- Final reconstructed video

---

## Custom GNU Radio Modules

The project uses custom GNU Radio Python modules, including:

ber_cal.py
data_separation.py
OAMP_Equalizer.py
Random_Modulation.py
video_receiver.py

These modules are used for:
- RM/OFDM data separation
- BER calculation
- OAMP equalization
- Random Modulation
- Video packet processing
- Video reconstruction

---

## Notes

Generated files such as BIN files, reconstructed images, reconstructed videos, cache files, and temporary outputs should normally not be committed to the repository.

Recommended .gitignore entries:

__pycache__/
*.pyc
*.bin
*.log

video_test/
picture/

rx_rm.png
rx_ofdm.png
rx_video_rm.mp4
rx_video_ofdm.mp4

---

## Status

The current implementation supports both image and video transmission using RM and OFDM.

The image pipeline includes automatic BIN generation and GUI-based testing.

The video pipeline supports complete packetization, transmission, reception, reconstruction, and performance evaluation.
