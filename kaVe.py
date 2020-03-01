import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv('train.csv')
#Preprocessing 
for  line,row in data.iterrows():
    if(data.loc[line,'folat']=="Mikrogram"):
        data.at[line, 'folat'] = "0.0001"
    if(data.loc[line,'water']=="Gram"):
        data.at[line, 'water'] = "0.0001"
data['calorie'] = pd.to_numeric(data['calorie'],errors='coerce')
data['water'] = pd.to_numeric(data['water'],errors='coerce')
data['carbohydrate']=pd.to_numeric(data['carbohydrate'],errors='coerce')
data['fiber'] = pd.to_numeric(data['fiber'],errors='coerce')
data['sugar'] = pd.to_numeric(data['sugar'],errors='coerce')
data['protein'] = pd.to_numeric(data['protein'],errors='coerce')
data['fat'] = pd.to_numeric(data['fat'],errors='coerce')
data['sfat'] = pd.to_numeric(data['sfat'],errors='coerce')
data['cholesterol'] = pd.to_numeric(data['cholesterol'],errors='coerce')
data['sodium'] = pd.to_numeric(data['sodium'],errors='coerce')
data['potassium'] = pd.to_numeric(data['potassium'],errors='coerce')
data['calcium'] = pd.to_numeric(data['calcium'],errors='coerce')
data['vit_A'] = pd.to_numeric(data['vit_A'],errors='coerce')
data['vit_C'] = pd.to_numeric(data['vit_C'],errors='coerce')
data['vit_D'] = pd.to_numeric(data['vit_D'],errors='coerce')
data['vit_E'] = pd.to_numeric(data['vit_E'],errors='coerce')
data['vit_K'] = pd.to_numeric(data['vit_K'],errors='coerce')
data['vit_B6'] = pd.to_numeric(data['vit_B6'],errors='coerce')
data['vit_B12'] = pd.to_numeric(data['vit_B12'],errors='coerce')
data['thiamin'] = pd.to_numeric(data['thiamin'],errors='coerce')
data['riboflavin'] = pd.to_numeric(data['riboflavin'],errors='coerce')
data['niacin'] = pd.to_numeric(data['niacin'],errors='coerce')
data['iron'] = pd.to_numeric(data['iron'],errors='coerce')
data['folat'] = pd.to_numeric(data['folat'],errors='coerce')
data['pantotenik'] = pd.to_numeric(data['pantotenik'],errors='coerce')
data['fosfor'] = pd.to_numeric(data['fosfor'],errors='coerce')
data['magnesium'] = pd.to_numeric(data['magnesium'],errors='coerce')
data['cinko'] = pd.to_numeric(data['cinko'],errors='coerce')
data['copper'] = pd.to_numeric(data['copper'],errors='coerce')
data['selenium'] = pd.to_numeric(data['selenium'],errors='coerce')
data['manganese'] = pd.to_numeric(data['manganese'],errors='coerce')

data_mean=data.fillna(data.mean())

x=data_mean.iloc[:,1:32].values
y=data_mean.iloc[:,32:33].values
rf_reg=RandomForestClassifier(n_estimators=15,random_state=500)
rf_reg.fit(x,y.ravel())
k_prediction=rf_reg.predict(x)
dpk=pd.DataFrame(k_prediction)
#Test
data_test = pd.read_csv('test.csv')
#Preprocessing
data_test['calorie'] = pd.to_numeric(data_test['calorie'],errors='coerce')
data_test['water'] = pd.to_numeric(data_test['water'],errors='coerce')
data_test['carbohydrate']=pd.to_numeric(data_test['carbohydrate'],errors='coerce')
data_test['fiber'] = pd.to_numeric(data_test['fiber'],errors='coerce')
data_test['sugar'] = pd.to_numeric(data_test['sugar'],errors='coerce')
data_test['protein'] = pd.to_numeric(data_test['protein'],errors='coerce')
data_test['fat'] = pd.to_numeric(data_test['fat'],errors='coerce')
data_test['sfat'] = pd.to_numeric(data_test['sfat'],errors='coerce')
data_test['cholesterol'] = pd.to_numeric(data_test['cholesterol'],errors='coerce')
data_test['sodium'] = pd.to_numeric(data_test['sodium'],errors='coerce')
data_test['potassium'] = pd.to_numeric(data_test['potassium'],errors='coerce')
data_test['calcium'] = pd.to_numeric(data_test['calcium'],errors='coerce')
data_test['vit_A'] = pd.to_numeric(data_test['vit_A'],errors='coerce')
data_test['vit_C'] = pd.to_numeric(data_test['vit_C'],errors='coerce')
data_test['vit_D'] = pd.to_numeric(data_test['vit_D'],errors='coerce')
data_test['vit_E'] = pd.to_numeric(data_test['vit_E'],errors='coerce')
data_test['vit_K'] = pd.to_numeric(data_test['vit_K'],errors='coerce')
data_test['vit_B6'] = pd.to_numeric(data_test['vit_B6'],errors='coerce')
data_test['vit_B12'] = pd.to_numeric(data_test['vit_B12'],errors='coerce')
data_test['thiamin'] = pd.to_numeric(data_test['thiamin'],errors='coerce')
data_test['riboflavin'] = pd.to_numeric(data_test['riboflavin'],errors='coerce')
data_test['niacin'] = pd.to_numeric(data_test['niacin'],errors='coerce')
data_test['iron'] = pd.to_numeric(data_test['iron'],errors='coerce')
data_test['folat'] = pd.to_numeric(data_test['folat'],errors='coerce')
data_test['pantotenik'] = pd.to_numeric(data_test['pantotenik'],errors='coerce')
data_test['fosfor'] = pd.to_numeric(data_test['fosfor'],errors='coerce')
data_test['magnesium'] = pd.to_numeric(data_test['magnesium'],errors='coerce')
data_test['cinko'] = pd.to_numeric(data_test['cinko'],errors='coerce')
data_test['copper'] = pd.to_numeric(data_test['copper'],errors='coerce')
data_test['selenium'] = pd.to_numeric(data_test['selenium'],errors='coerce')
data_test['manganese'] = pd.to_numeric(data_test['manganese'],errors='coerce')
data_test_mean=data_test.fillna(data_test.mean())
sample_submission=pd.read_csv('sample_submission.csv')
features=data_test_mean.iloc[:,1:32]
y_pred=rf_reg.predict(features)
y_pred=pd.DataFrame(y_pred)
sample_submission['ID']=sample_submission.index
sample_submission['Class']=y_pred
sample_submission.to_csv("DecisionTreeClassifier.csv",index=False)


