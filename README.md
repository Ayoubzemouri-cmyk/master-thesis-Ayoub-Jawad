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
- A **Drone Inventor kit**, used as the base mobility platform for the
  robot, providing the chassis, motors, and flight controller integration
  onto which the water quality sensors and ESP32 module were mounted.

## Repository Structure

```
water-quality-thesis/
├── report/            Final thesis PDF
├── slides/             Defense presentation slides
├── code/
│   ├── firmware/       ESP32 firmware (ThesisCode.ino)
│   ├── ai/              U-Net training script (ai_swamp_unet.py)
│   └── requirements.txt Python dependencies for the AI script
├── data/                Sample water quality measurements from field tests
├── drone-inventor/      Drone Inventor kit files (CAD / build guide / parts)
├── hardware/            Wiring diagrams and sensor connection schematics
└── results/             Plots, calibration curves, and evaluation metrics
```

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
  mobility platform (CAD files, build guide, and part list)
- `hardware/` — Wiring diagrams and sensor connection schematics
- `results/` — Plots, calibration curves, U-Net evaluation metrics
  (per-class IoU, confusion matrix), and field test results

## Reproducing the Experiments

### Firmware (ESP32)

1. Open `code/firmware/ThesisCode.ino` in the **Arduino IDE**.
2. Install required libraries via the Arduino Library Manager:
   - MAVLink library
   - Atlas Scientific EZO sensor library
   - WiFi/BLE libraries (bundled with the ESP32 board package)
3. Select board: **ESP32** (ESP32-WROOM-32).
4. Select the correct COM port for your ESP32 and upload the sketch.

### AI Vision Model (U-Net terrain segmentation)

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/water-quality-thesis.git
   cd water-quality-thesis/code/ai
   ```

2. Install the Python dependencies:

   ```bash
   pip install -r ../requirements.txt
   ```

3. Download the datasets used for training:
   - **TU Graz Semantic Drone Dataset**: [https://www.tugraz.at/index.php?id=22387](https://ivc.tugraz.at/research-project/semantic-drone-dataset/)
   - **FloodNet Track-1 Dataset**: https://github.com/BinaLab/FloodNet-Supervised_v1.0

   Place the downloaded datasets in a `datasets/` folder alongside the
   script, or update the paths at the top of `ai_swamp_unet.py` to point
   to your local dataset location.

4. Run the training script:

   ```bash
   python ai_swamp_unet.py
   ```

5. Evaluation metrics (per-class IoU, confusion matrix) and sample
   predictions will be saved to the `results/` folder after training
   completes.

## Data

The `data/` folder contains sample water quality measurements (pH,
dissolved oxygen, temperature, conductivity) collected during field
tests. The full dataset is available on request — contact the authors.

## Citation

If you use this work, please cite it as:

```
Zemouri, A., & Zennaki, M. J. (2026). Development of an Autonomous Water
Quality Assessment Robot for Swampy Water [Master's thesis]. Vrije
Universiteit Brussel, ELO-ICT.
```

## Authors & Supervision

- **Ayoub Zemouri** — Vrije Universiteit Brussel (ELO-ICT)
- **Mohamed Jawad Zennaki** — Vrije Universiteit Brussel (ELO-ICT)
- Supervisors: **Abdellah Touhafi**, **Souhail Fatimi**
