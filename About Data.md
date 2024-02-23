# About Dataset

This data set contains termal videos and recordings of yarns for photothermal actuation measurements. Zahraalsadat Madani produced and extruded the filaments, Maija Vaara and Laura Koskelo twisted and winded the filaments, Susobhan Das, and Camilo Arias made the photoactivation recordings and Pedro Silva made the measurements. Samples names can be found in the table below and measurements were performed not sequentialy. 
### Folders, Files and file formats

If the data includes images or audio, you can mention the file format eg.(.svg, .png, .mpeg).
```
LTCAM:
- 14 npy pixel data with temperatures (mat files)
- 14 npy time data (time files)
- Supplementary images of samples and experimental setup

Butterfly:
- 2 npz pixel with temperatures and time data
- 2 mp4 recordings of the activation of wings

Rotating:
- 6 npz pixel with temperatures and time data
- 6 mp4 recordings of activation of samples in a rotating platform
```

#### Optical parameters setup for 'LTCAM' experiments

- **Laser specification**: Ultrafast laser with pulse width 230 fs and repetition rate 2kHz. (Spectra-Physics, TOPAS)
- Incident wavelength:  800nm (Horizontal polarization)
- Beam Diameter: 5 mm
- Average Power: tunable based on experimental need
- Beam Diameter is larger than the width of the sample. Therefore, to calculate the effective incident power, it has to be the fraction of the total power based on the area of the sample exposed. 
- **Lens System**:  A part of experiment is done under the focused light condition with a lens of 100mm focal length. 
- **Power Meter**: Thorlab Power meter (Model No. S401C) is used to measure the incident power on the sample.(Wavelength range: 190nm to 20000nm). 
- Setup photo:
![[PXL_20230414_081710929.jpg]]

#### Optical parameters setup for 'Butterfly' and 'Rotating' experiments

- Laser: Cobolt 06-MLD | 808 nm | max power = 100 mW
- Setup photo:
# About Dataset

This data set contains termal videos and recordings of yarns for photothermal actuation measurements. Zahraalsadat Madani produced and extruded the filaments, Maija Vaara and Laura Koskelo twisted and winded the filaments, Susobhan Das, and Camilo Arias made the photoactivation recordings and Pedro Silva made the measurements. Samples names can be found in the table below and measurements were performed not sequentialy. 
### Folders, Files and file formats

If the data includes images or audio, you can mention the file format eg.(.svg, .png, .mpeg).
```
LTCAM:
- 14 npy pixel data with temperatures (mat files)
- 14 npy time data (time files)
- Supplementary images of samples and experimental setup

Butterfly:
- 2 npz pixel with temperatures and time data
- 2 mp4 recordings of the activation of wings

Rotating:
- 6 npz pixel with temperatures and time data
- 6 mp4 recordings of activation of samples in a rotating platform
```

#### Optical parameters setup for 'LTCAM' experiments

- **Laser specification**: Ultrafast laser with pulse width 230 fs and repetition rate 2kHz. (Spectra-Physics, TOPAS)
- Incident wavelength:  800nm (Horizontal polarization)
- Beam Diameter: 5 mm
- Average Power: tunable based on experimental need
- Beam Diameter is larger than the width of the sample. Therefore, to calculate the effective incident power, it has to be the fraction of the total power based on the area of the sample exposed. 
- **Lens System**:  A part of experiment is done under the focused light condition with a lens of 100mm focal length. 
- **Power Meter**: Thorlab Power meter (Model No. S401C) is used to measure the incident power on the sample.(Wavelength range: 190nm to 20000nm). 
- Setup photo:
![[PXL_20230414_081710929.jpg]]

#### Optical parameters setup for 'Butterfly' and 'Rotating' experiments

- Laser: Cobolt 06-MLD | 808 nm | max power = 100 mW
- Setup photo:
![[Drawing 2023-03-17 15.47.09.excalidraw|600]]
#### LTCAM Samples details

![[all_v2.png]]

| Name in the article    | Sample No. | Mandrel Dia (mm) | Original length (cm) | Twisting (rounds) | Twists/cm | Twisting Direction | coiling direction |
| --- | ---------- | ---------------- | -------------------- | ----------------- | --------- | ------------------ | ----------------- |
| ZSØ1A    | M733       | 1                | 50                   | 260               | 5.2       | z                  | s                 |
| ZSØ2A    | M734       | 2                | 50                   | 301               | 6.02      | z                  | s                 |
| ZSØ1B    | M781       | 1                | 50                   | 258               | 5.16      | z                  | s                 |
| ZSØ1C    | M782       | 1                | 50                   | 262               | 5.24      | z                  | s                 |
| ZSØ2B    | M783       | 2                | 48                   | 270               | 5.63      | z                  | s                 |
| ZSØ2C    | M784       | 2                | 50                   | 227               | 4.54      | z                  | s                 |
| ZZØ2A    | M785       | 2                | 50                   | 339               | 6.78      | s                  | s                 |
| ZZØ2B    | M786       | 2                | 50                   | 226               | 4.52      | s                  | s                 |

### Experiments 'LTCAM'

| Filename      | Sample name | Laser power (mW) | Time ON (s) | Time OFF (s) | Cycles |
|---------------|-------------|------------------|-------------|--------------|--------|
| 230414_112622 | ZSØ2A       | 50               | 60          | 30           | 1      |
| 230414_112851 | ZSØ2A       | 100              | 60          | 30           | 1      |
| 230414_113149 | ZSØ2A       | 150              | 60          | 30           | 1      |
| 230414_113420 | ZSØ2A       | 200              | 60          | 30           | 1      |
| 230414_113656 | ZSØ2A       | 250              | 60          | 30           | 1      |
| 230414_113955 | ZSØ2A       | 300              | 60          | 30           | 1      |
| 230414_114838 | ZSØ1B       | 300              | 30          | 30           | 3      |
| 230414_115518 | ZSØ1C       | 300              | 30          | 30           | 3      |
| 230414_120115 | ZSØ2B       | 300              | 30          | 30           | 3      |
| 230414_120714 | ZSØ2C       | 300              | 30          | 30           | 3      |
| 230414_121714 | ZZØ2B       | 300              | 30          | 30           | 3      |
| 230414_122255 | ZZØ2A       | 300              | 30          | 30           | 3      |
| 230414_122808 | ZSØ2A       | 300              | 30          | 30           | 3      |
| 230414_123244 | ZSØ1A       | 300              | 30          | 30           | 3      |

### Experiments 'Butterfly'

| Filename (Recording) | Filename (IR video) | Laser power (mW) | Time ON (s) | Time OFF (s) | Cycles | Obs.                  |
|----------------------|---------------------|------------------|-------------|--------------|--------|-----------------------|
| 20231002_134454162   | 231002_164924       | 100              | 1           | 1            | 100    | Left wing activation  |
| 20231002_135842563   | 231002_170312       | 100              | 1           | 1            | 100    | Right wing activation |

### Experiments 'Rotating'

| Filename (Recording) | Filename (IR video) | Laser power (mW) | Time ON (s) | Rotation deg/s (measured) |
|----------------------|---------------------|------------------|-------------|---------------------------|
| 20231003_073512244   | 231003_103940       | 100              | 120         | -28.0                     |
| 20231003_073755982   | 231003_104223       | 100              | 120         | -6.6                      |
| 20231003_074535766   | 231003_105003       | 100              | 120         | -15.3                     |
| 20231003_085141041   | 231003_115608       | 100              | 120         | 5.2                       |
| 20231003_085417756   | 231003_115845       | 100              | 120         | 27.3                      |
| 20231003_085716949   | 231003_120144       | 100              | 120         | 12.8                      |
#### LTCAM Samples details

![[all_v2.png]]

| Name in the article    | Sample No. | Mandrel Dia (mm) | Original length (cm) | Twisting (rounds) | Twists/cm | Twisting Direction | coiling direction |
| --- | ---------- | ---------------- | -------------------- | ----------------- | --------- | ------------------ | ----------------- |
| ZSØ1A    | M733       | 1                | 50                   | 260               | 5.2       | z                  | s                 |
| ZSØ2A    | M734       | 2                | 50                   | 301               | 6.02      | z                  | s                 |
| ZSØ1B    | M781       | 1                | 50                   | 258               | 5.16      | z                  | s                 |
| ZSØ1C    | M782       | 1                | 50                   | 262               | 5.24      | z                  | s                 |
| ZSØ2B    | M783       | 2                | 48                   | 270               | 5.63      | z                  | s                 |
| ZSØ2C    | M784       | 2                | 50                   | 227               | 4.54      | z                  | s                 |
| ZZØ2A    | M785       | 2                | 50                   | 339               | 6.78      | s                  | s                 |
| ZZØ2B    | M786       | 2                | 50                   | 226               | 4.52      | s                  | s                 |

### Experiments 'LTCAM'

| Filename      | Sample name | Laser power (mW) | Time ON (s) | Time OFF (s) | Cycles |
|---------------|-------------|------------------|-------------|--------------|--------|
| 230414_112622 | ZSØ2A       | 50               | 60          | 30           | 1      |
| 230414_112851 | ZSØ2A       | 100              | 60          | 30           | 1      |
| 230414_113149 | ZSØ2A       | 150              | 60          | 30           | 1      |
| 230414_113420 | ZSØ2A       | 200              | 60          | 30           | 1      |
| 230414_113656 | ZSØ2A       | 250              | 60          | 30           | 1      |
| 230414_113955 | ZSØ2A       | 300              | 60          | 30           | 1      |
| 230414_114838 | ZSØ1B       | 300              | 30          | 30           | 3      |
| 230414_115518 | ZSØ1C       | 300              | 30          | 30           | 3      |
| 230414_120115 | ZSØ2B       | 300              | 30          | 30           | 3      |
| 230414_120714 | ZSØ2C       | 300              | 30          | 30           | 3      |
| 230414_121714 | ZZØ2B       | 300              | 30          | 30           | 3      |
| 230414_122255 | ZZØ2A       | 300              | 30          | 30           | 3      |
| 230414_122808 | ZSØ2A       | 300              | 30          | 30           | 3      |
| 230414_123244 | ZSØ1A       | 300              | 30          | 30           | 3      |

### Experiments 'Butterfly'

| Filename (Recording) | Filename (IR video) | Laser power (mW) | Time ON (s) | Time OFF (s) | Cycles | Obs.                  |
|----------------------|---------------------|------------------|-------------|--------------|--------|-----------------------|
| 20231002_134454162   | 231002_164924       | 100              | 1           | 1            | 100    | Left wing activation  |
| 20231002_135842563   | 231002_170312       | 100              | 1           | 1            | 100    | Right wing activation |

### Experiments 'Rotating'

| Filename (Recording) | Filename (IR video) | Laser power (mW) | Time ON (s) | Rotation deg/s (measured) |
|----------------------|---------------------|------------------|-------------|---------------------------|
| 20231003_073512244   | 231003_103940       | 100              | 120         | -28.0                     |
| 20231003_073755982   | 231003_104223       | 100              | 120         | -6.6                      |
| 20231003_074535766   | 231003_105003       | 100              | 120         | -15.3                     |
| 20231003_085141041   | 231003_115608       | 100              | 120         | 5.2                       |
| 20231003_085417756   | 231003_115845       | 100              | 120         | 27.3                      |
| 20231003_085716949   | 231003_120144       | 100              | 120         | 12.8                      |