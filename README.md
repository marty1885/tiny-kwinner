#tiny-kwinner
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

## Reference
* [How Can We Be So Dense? The Benefits of Using Highly Sparse Representations](https://arxiv.org/abs/1903.11257) - Numenta, 2019

