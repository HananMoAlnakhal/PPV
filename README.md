# PPV
this is a regression model using Pycaret
# resources:
- [PyCaret Regression Tutorial](https://nbviewer.org/github/pycaret/pycaret/blob/master/tutorials/Tutorial%20-%20Regression.ipynb)
- [PyCaret 2 Regression Example](https://nbviewer.org/github/pycaret/examples/blob/main/PyCaret%202%20Regression.ipynb)
- [Prof. Pedram Jahangiry](https://colab.research.google.com/github/PJalgotrader/platforms-and-tools/blob/main/PyCaret/PyCaret-RegressionDemo.ipynb#scrollTo=BVaWoaM1-sWJ)

# 1. Installing +importing libraries
`!pip install pycaret[full]`
```python
import pandas as pd
import pycaret
from pycaret.regression import *
```
# 2. Imported the data using the [URL](https://docs.google.com/spreadsheets/d/1LWurKSC0UOMrv_AdmCaIHZllNll__HU13oFJsm0_siY/export?format=csv)
```python
df = pd.read_csv(url)
```
# 3. splited data into seen and unseen 
> used ->90%(will be used in ) to unused ->10%

> splited the data after shuffling it because I noticed that the first half of the data is Male and the other half is Female and if we split it as it is it will be biased because it will be trained mostly on males

# **steps** to create:
## 4. used the seen data in the setup
```python
exp = RegressionExperiment()
exp.setup(data, target = 'PPV', session_id = 123,ordinal_features=None,preprocess=False)
```
## 5. then used `exp.compare_models(sort='RMSE')` to get the best models 
## 6. based on the results chosen ***lightgbm Light Gradient Boosting Machine***
![image](https://github.com/user-attachments/assets/1494c74b-d47a-4a70-bde8-f6a4b928f5df)

## 7. made the model 
> using this code `lightgbm = exp.create_model('lightgbm', fold=5)`
## 8. Made essential plots 

![residuals](https://github.com/user-attachments/assets/628530ed-b686-496f-b833-6d8d9986fa5b)
residuals

![model error](https://github.com/user-attachments/assets/13b6c4c5-428f-4584-83f3-551d21ec93bb)
model error

![learning cuve](https://github.com/user-attachments/assets/e5fe0eca-7b86-423a-814e-823865e7ebf9)
learning curve

![Validation curve](https://github.com/user-attachments/assets/e31999cb-6617-413a-b52f-06e023effe99)
Validation curve

![feature importance](https://github.com/user-attachments/assets/be2fa200-d311-48f6-a503-afe1508cb57a)
feature importance

## 9. saw the tuned model 
> (the tuned model turned out to be worse btw)

## 10. tested the model 
> compared between the `PPV` and	`prediction_label`

 they are so close!
 | Gender|	Age	|Dur	|PPV	|prediction_label|
 |-------|------|-----|-----|----------------|
 |Female|40|4|3.820880|3.826051|
 |Male	|1|35|21.815239|21.812535|
 |Female|52|5|4.679705|4.671349|
 |Male	|46|17|12.891843|12.773545|

## 11. finalized the model
## 12. tested the model on unseen data! (the 10%)

## 13. saved the model as PPV.pkl

# to see the work:
> Download the `PPV.pkl` file and load it into your env

>or explore the notebook `ML_Reg.ipynb`
