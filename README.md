## Setup for Phase 2

 **Pre-Requisite:** Make sure you have the training data folder `data` in the directory `Group13_p1/Phase2/`, and place the `P1Ph2TestSet` in the directory `Group13_p1/Phase2/Data/Test/`. Also install the checkpoints from the links below and place it in the directory `Group13_p1/Phase2/Checkpoints/`.

**Supervised Checkpoint Link:** https://wpi0-my.sharepoint.com/:u:/g/personal/sporwal_wpi_edu/EbEvSchMS1ZEhTQsHcSssyMBwCEfUPRJj4hlhCRAXXg73A?e=wx24CM

**Unsupervised Checkpoint Link:** https://wpi0-my.sharepoint.com/:u:/g/personal/sporwal_wpi_edu/ET-y14jv_gFMuPxcd8c9RKwB6ASryu3dcpYgcfTp1e2xgg?e=7Y4xU6

1. Steps to generate synthetic data:
*Note:* These commands will generate a small dataset for evaluation purposes. We used 500000 images for training but this will just generate 10000 images.
```bash
cd Group13_p1/Phase2/Code 
python3 GenerateTrainData.py
python3 GenerateValData.py
```

2. a) Steps to run Train.py using **Supervised** Approach:
```bash
python3 Train.py --ModelType="Sup"
```
2. b) Steps to run Train.py using **Unsupervised** Approach:
```bash
python3 Train.py --ModelType="Unsup"
```
3. Steps to run Test.py:
```bash
python3 Test.py --ImageCount=5 
```
*Note:* This will generate **Code/OutputImages/TestImages/** folder where the results will be saved.

4. a) Steps to run Wrapper.py using **Supervised** Approach:
```bash
python3 Wrapper.py --ModelType="Sup" --NameOfTestSet="unity_hall"
```
4. b) Steps to run Wrapper.py using **Unsupervised** Approach:
```bash
python3 Wrapper.py --ModelType="Unsup" --NameOfTestSet="unity_hall"
```
*Note:* This will generate **Code/OutputImages/StitchedImages/** folder where the results will be saved.


## Setup for Phase 1

Before running the script, make sure to navigate to the correct directory and have all dependencies installed.

1. Navigate to the project directory:

```bash
cd Group13_p1/Phase1/Code 
```

## Running the Train Set (1,2,3)

```bash
python3 Wrapper.py --Set=Set1 --SetType=Train
```

## Running the Custom Set (1,2)

```bash
python3 Wrapper.py --Set=CustomSet1 --SetType=Train
```

## Running the Test Set (1,2,3,4)

```bash
python3 Wrapper.py --Set=TestSet1 --SetType=Test
```



