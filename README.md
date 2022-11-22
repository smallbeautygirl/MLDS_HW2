# MLDS-HW2-2022

### Source

- [Slide](https://docs.google.com/presentation/d/1ihPO_2WvHyrAHinLvGPXRs8jX8ED3Hri/edit?usp=sharing&ouid=112246576290315533201&rtpof=true&sd=true)
- [Dashboard](https://docs.google.com/spreadsheets/d/1Rx6OV64Br8EHrZpbjMa86rm9a8vsngNSjB12s4pO940/edit?usp=sharing)

### Idea

#### :one: Training

- :question: 訓練資料前處理

  - 讀取 training_data 裡面 1~8 月的 csv 檔案

  - 用 glob 的 function 把這八個月的 csv 整合為一個 csv

  - 利用 pandas 來讀取剛剛整合完的 training data csv，**並把 "time" 設為 index**

  - 特徵縮放-將特徵下的資料值設定在相同的範圍，解決資料分布範圍不一致的問題
    - `MinMaxScaler`

- :question: 使用什麼模型或方法

  - model  :point_right:  **The Sequential model**

  - layers :point_down: 

    - LSTM

    - Dense (輸出設為 24 :point_right: 每個小時只做一個動作)

  - model summary:
    - ![image](model-summary.png)
  
- :question:  如何訓練模型

  - epoch = 1 :point_right: 使用 cpu 訓練的
  - epoch = **60** :point_right: 使用 **gpu** 訓練的
    - environment
    - ![image](nvidia-smi.png)

#### :two: Validation

- :question: 如何藉由媒合結果改善程式

  - action:

    - buy 情境:

      - 產電量小於所需用電量

    - sell 情境:

      - 產電量大於所需用電量
      - 由於 **早上 10:00 至 下午 3:00** 產電量較高

  - price:

    - buy 情境:
      - 買的價格限制**比台電低**

    - sell 情境:
      - 賣的價格限制比**比台電低**