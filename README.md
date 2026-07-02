# Water Quality Assessment Robot

## Purpose
This repository contains the code, data samples, hardware documentation,
report, and presentation slides for the master's thesis, "**Development
of an Autonomous Water Quality Assessment Robot for Swampy Water**," by
**Ayoub Zemouri & Mohamed Jawad Zennaki**, completed at **VUB
(Vrije Universiteit Brussel), ELO-ICT**, under the supervision of
**Abdellah Touhafi & Souhail Fatimi**.

The thesis presents the design, development, and evaluation of an
autonomous robot capable of measuring water quality parameters (pH,
dissolved oxygen, temperature, conductivity) in swampy and hard-to-access
water regions. The system combines:
- An **ESP32-based firmware** that triggers Atlas Scientific sensor
  sampling cycles via MAVLink messages from the flight/navigation
  controller.
- A **U-Net semantic segmentation model** trained to classify terrain
  as safe/unsafe for the robot to navigate, using the TU Graz Semantic
  Drone Dataset and the FloodNet Track-1 dataset.
- A **[Drone Inventor kit / platform — describe briefly]** used as the
  base mobility platform for the robot.

## Repository Structure
- `report/` — Final thesis PDF (`MA_IW_EM_Zennaki_Zemouri_Mohamed_Ayoub_S2_2526.pdf`)
- `slides/` — Defense presentation slides (`MA_THESIS_PRES.pdf`)
- `code/`
  - `firmware/ThesisCode.ino` — ESP32 firmware handling MAVLink-triggered
    Atlas Scientific water quality sampling cycles
  - `ai/ai_swamp_unet.py` — U-Net training script: combines the TU Graz
    Semantic Drone Dataset and FloodNet Track-1 dataset into a binary
    safe/unsafe corpus, trains a U-Net (32-64-128-256 encoder,
    512-channel bottleneck, ~7.77M parameters) for 100 epochs, and
    evaluates it on a held-out test set
  - `requirements.txt` — Python dependencies for the AI script
    (TensorFlow/Keras, etc.)
- `data/` — Sample water quality measurements collected during field
  tests (full dataset available on request)
- `drone-inventor/` — Drone Inventor kit files used for the robot's
  mobility platform ([CAD files / build guide / part list — specify])
- `hardware/` — Wiring diagrams and sensor connection schematics
- `results/` — Plots, calibration curves, U-Net evaluation metrics
  (per-class IoU, confusion matrix), and field test results

## Reproducing the Experiments

### Firmware (ESP32)
1. Open `code/firmware/ThesisCode.ino` in the **Arduino IDE**.
2. Install required libraries: **[list them, e.g. MAVLink library,
   Atlas Scientific EZO library, WiFi/BLE libraries used]**
3. Select board: **ESP32** (specify exact dev board model, e.g.
   ESP32-WROOM-32).
4. Connect the Atlas Scientific sensor probes as shown in
   `hardware/wiring_diagram.png`.
5. Upload the sketch. The firmware listens for MAVLink messages from
   the flight/navigation controller and triggers a sampling cycle on
   the Atlas Scientific sensors when received.

### AI Vision Model (U-Net terrain segmentation)
1. Clone this repository:
