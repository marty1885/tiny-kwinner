# tiny-kwinner

A simple implementation of the [k-winner algorithm](ihttps://arxiv.org/abs/1903.11257) in tiny-dnn. 

## How to build

```
git clone https://github.com/marty1885/tiny-kwinner
g++ main.cpp -o main -Ofast -ltbb -march=native
```

To run the program, please download the MNIST to dataset to the current folder.

```
./main
```

`main.cpp` is the original version without boosting. `main2.cpp` is with boosting a different method adding noise to image.

## Requirments
* An AVX capable CPU - Intel Sandy Bridge, AMD Jaguar/Bulldozer or later
If you are running on loder x86 processors or not on x86. Plase comment 
the line `#define CNN_USE_AVX` in main.cpp. But do note that this drastically 
decreases performance.

* Intel TBB
* C++17 capcable compiler
* [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn)

## Results

Performance of kwinner vs other regularizarion metheds.

| Noise/Accuracy(%) |  Raw   |  Dropout |  Batch Norm |  k-winner | 
|-------------------|--------|----------|-------------|-----------| 
| 0                 | _98.17_|  95.73   |  93.08      |**95.81**  | 
| 0.05              | _97.72_|**95.26** |  92.15      |  95.09    | 
| 0.1               | _96.74_|**94.15** |  91.81      |  94.08    | 
| 0.2               | _93.48_|  92.06   |  88.56      |**92.07**  | 
| 0.3               |  87.31 |  86.48   |  80.63      |**88.53**  | 
| 0.4               |  77.19 |  78.55   |  65.13      |**83.2**   | 
| 0.5               |  63.35 |  67.84   |  47.05      |**76.17**  | 
| 0.6               |  45.56 |  55.61   |  28.36      |**65.7**   | 
| 0.7               |  29.93 |  41.78   |  15         |**51.79**  | 
| 0.8               |  18.49 |  28.21   |  10.67      |**34.82**  | 

With boosting (and adding gaussian noise instead of setting random value to pixels):

| noise/accuracy(%) | Raw   | Dropout | Batch Norm | KWinner | KWinner+boosting | 
|-------------------|-------|---------|------------|---------|------------------| 
| 0.0               | 98.91 | 98.11   | 97.79      | 98.47   | 98.22            | 
| 0.05              | 94.58 | 76.99   | 62.23      | 71.35   | **95.68**         | 
| 0.1               | 94.33 | 76.97   | 61.77      | 70.89   | **95.55**         | 
| 0.2               | 93.4  | 76.39   | 60.6       | 68.28   | **95.35**         | 
| 0.3               | 91.29 | 75.69   | 58.73      | 66.03   | **94.81**         | 
| 0.4               | 88.58 | 74.25   | 56.49      | 62.31   | **94.15**         | 
| 0.5               | 85.57 | 71.75   | 53.05      | 57.43   | **92.66**         | 
| 0.6               | 82.28 | 68.25   | 48.23      | 51.25   | **88.96**         | 
| 0.7               | 77.52 | 63.22   | 42.65      | 43.39   | **82.41**         | 
| 0.8               | 71.72 | 53.76   | 32.94      | 35.75   | **74.18**         | 


## Reference
* [How Can We Be So Dense? The Benefits of Using Highly Sparse Representations](https://arxiv.org/abs/1903.11257) - Numenta, 2019

