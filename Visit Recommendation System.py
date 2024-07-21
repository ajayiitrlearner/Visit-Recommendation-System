#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Intitial Data Sourcing, Cleaning , Modifying steps are removed to secure those APIs and as data is confidential


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Final_Data,Target,test_size=0.2, random_state=42)


# In[ ]:


Final_Data.columns


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


ER_Names_Train,ER_Names_Test=train_test_split(ER_Names,test_size=0.2, random_state=42)


# # Randomforest Training

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# In[ ]:


import gc
gc.collect()


# In[ ]:


model=RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=6,n_jobs=-1, random_state=42, verbose=1)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Number of trees in random forest
n_estimators = [200,400,600]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

rf_Model=RandomForestClassifier()

param_comb = 3

rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid,n_iter=param_comb, cv = 5, verbose=10, n_jobs = -1)


# In[ ]:


rf_RandomGrid.fit(X_train,y_train)


# In[ ]:


rf_RandomGrid.best_estimator_


# In[ ]:


best_RF_model=RandomForestClassifier(max_depth=4, max_features='sqrt',n_estimators=200,verbose=0,n_jobs=-1)


# In[ ]:


best_RF_model.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support,accuracy_score)


# In[ ]:


X_test.shape


# In[ ]:


ypreds_test  = best_RF_model.predict(X_test)


# In[ ]:


ypreds_test_proba  = best_RF_model.predict_proba(X_test)


# In[ ]:


ypreds_test_proba.shape,X_test.shape


# In[ ]:


ypreds_test_proba[0]


# In[ ]:


best_RF_model.classes_


# In[ ]:


Top_K=50
top_k_recom_prob=[]
for i in range(X_test.shape[0]):
    ypreds_test_proba_top_k=(-ypreds_test_proba[i]).argsort()[:Top_K]# indices
    cls=[]
    for j in ypreds_test_proba_top_k:
        cls.append(best_RF_model.classes_[j])
    top_k_recom_prob.append(cls)


# In[ ]:


len(top_k_recom_prob)


# In[ ]:


top_k_recom_prob[:5]


# In[ ]:


import numpy as np
def Mean_Accuracy(y_true_list, y_reco_list, users):
    Accuracy_all = list()
    for u in range(users.shape[0]):
        y_true = y_true_list[u]
        y_reco = y_reco_list[u]
        common_items = set(y_reco).intersection(y_true)
        if len(common_items)==0:
            acc=0
        else:
            acc=1
        Accuracy_all.append(acc)
    return np.mean(Accuracy_all)


# In[ ]:


print("Top 50 Test Accuracy",Mean_Accuracy(y_test.values, top_k_recom_prob, X_test))


# In[ ]:


X_train.shape[0]


# In[ ]:


ypreds_test_proba  = best_RF_model.predict_proba(X_train)


# In[ ]:


ypreds_test  = best_RF_model.predict(X_test)


# In[ ]:


Top_K=50
top_k_recom_prob=[]
for i in range(X_train.shape[0]):
    ypreds_test_proba_top_k=(-ypreds_test_proba[i]).argsort()[:Top_K]# indices
    cls=[]
    for j in ypreds_test_proba_top_k:
        cls.append(best_RF_model.classes_[j])
    top_k_recom_prob.append(cls)


# In[ ]:


print("Top 50 Train Accuracy",Mean_Accuracy(y_train.values, top_k_recom_prob, X_train))


# In[ ]:


ypreds_train=best_RF_model.predict(X_train)


# In[ ]:


accuracy = accuracy_score(y_train, ypreds_train)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


accuracy = accuracy_score(y_test, ypreds_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


print(classification_report(y_train,ypreds_train))


# In[ ]:


print(classification_report(y_test,ypreds_test))


# # Model Saving

# In[ ]:


import joblib

model_save_path='./python_data/models/'
#joblib.dump(best_RF_model, model_save_path+'RF_VRM_20_04_2023.pkl')
#joblib.dump(best_RF_model, model_save_path+'RF_VRM_12_12_2023.pkl')


# # Model Loading

# In[147]:


import joblib

model_save_path='./python_data/models/'
model=joblib.load( model_save_path+'RF_VRM_12_12_2023.pkl')


# # Class Recommendations

# In[148]:


class Recommendation_System:
    def __init__(self, model,K_recommends):
        self.model=model
        self.K_recommends=K_recommends
        
    def Top_K_Rec(rec_list,K_recommends,SOL_IDs_Allocated_to_ER_ID):
        final_dic={k: v for k, v in sorted(rec_list.items(), key=lambda item: item[1],reverse=True)}
        solid_numbers=list(final_dic.keys())# sorted solid based on high prob score
        score=list(final_dic.values())
        #print("Top",K,"Recommendations")
        TOP_K=[]
        for index in range(len(solid_numbers)):
            if solid_numbers[index] in SOL_IDs_Allocated_to_ER_ID:#check condition for solid allocated
                TOP_K.append(solid_numbers[index])   
            if len(TOP_K)==K_recommends:
                break
        return TOP_K 
    
        #############################################################################################################
    def Solid_recommendations(ERName_Test,testing_data,SOL_ID_list_asper_RF_model,SOL_IDs_Allocated_to_ER_ID):
        #INPUT
        ERName_Test=ERName_Test[['ER ID']]# using er id and not er name
        ERName_Test_LIST=[]
        Recommendation_List=[]
        for each_user_index in tqdm(range(ERName_Test.shape[0])):
            user_sample=testing_data[each_user_index:each_user_index+1] #current user features
            Name=ERName_Test.iloc[each_user_index][0] # Current user UCIC value
            #customers features input
            # solid recommendations
            prediction_prob=model.predict_proba(user_sample)[0]
            tup_logo_score_dic={} #empty dic
            #all logo with prob
            for l_index in range(0,len(SOL_ID_list_asper_RF_model)):

                solid_number=SOL_ID_list_asper_RF_model[l_index]
                tup_logo_score_dic[str(solid_number)]=prediction_prob[l_index]

            #OUTPUT
            K=K_recommends #no of recomm
            TOP_K=Top_K_Rec(tup_logo_score_dic,K_recommends,SOL_IDs_Allocated_to_ER_ID) # Top K solid outputs
            #saving results for current Name
            ERName_Test_LIST.append(Name)
            Recommendation_List.append(TOP_K)
        return ERName_Test_LIST,Recommendation_List
    ##################################################################################  
    def Custom_Recommend_SOLID(model,K_recommends,Training_Data,ER_ID,solid_VRM_ACTIVE_df):
        Training_Data.rename(columns = {'ER_ID':'ER ID'}, inplace = True)
        Training_Data.rename(columns = {'ER_NAME':'ER NAME'}, inplace = True)
        print(Training_Data.columns)
        ER_Names=Training_Data[['ER ID','ER NAME']]
        Training_Data.drop(['SOLID','ER ID','ER NAME'],axis=1,inplace=True)
        Features=Training_Data
        Features['LAST DATE OF VISIT'] = Features['LAST DATE OF VISIT'].replace(['NOT_VISITED'], 9999)
        dates=Features['LAST DATE OF VISIT'].values
        today = datetime.date.today()
        for index in range(0,Features.shape[0]):
            if dates[index]!=9999:
                date_time_str = dates[index]
                date_time_obj = datetime.datetime.strptime(date_time_str, '%d %b %Y')
                date_time_obj = date_time_obj.date()
                diff=today-date_time_obj
                dates[index]=int(diff.days)
        Features.drop(['CITY','STATE','BRANCH'],axis=1,inplace=True)  
        Features['LAST DATE OF VISIT']=Features['LAST DATE OF VISIT'].astype(int)
        objcolumnsName = Features.columns[Features.dtypes == 'object']
        Final_Data=pd.get_dummies(Features, columns=objcolumnsName)
        #print(Final_Data.shape)
        Final_Data.insert(loc=0, column='ER ID', value=ER_Names['ER ID'].values)
        Final_Data.insert(loc=1, column='ER Name', value=ER_Names['ER NAME'].values)
        Final_Data.drop_duplicates(subset=['ER ID','ER Name'], keep="last",inplace=True)

        Final_Data=Final_Data[Final_Data['ER ID']==str(ER_ID)]#selecting data for given ER_ID
        SOL_IDs_Allocated_to_ER_ID= list(solid_VRM_ACTIVE_df[solid_VRM_ACTIVE_df['ER ID']==str(ER_ID)]['SOL_ID'].values)

        ERName_Test=Final_Data[['ER ID','ER Name']]
        Final_Data=Final_Data.drop(['ER ID','ER Name'],axis=1)
        SOL_ID_list_asper_RF_model=model.classes_
        ERName_Test_LIST,Recommendation_List=Solid_recommendations(model,K_recommends,ERName_Test,Final_Data,\
                                                                   SOL_ID_list_asper_RF_model,SOL_IDs_Allocated_to_ER_ID)
        return ERName_Test,Recommendation_List    


# # Custom Recommendations

# In[149]:


def Top_K_Rec(rec_list,K,SOL_IDs_Allocated_to_ER_ID):
    final_dic={k: v for k, v in sorted(rec_list.items(), key=lambda item: item[1],reverse=True)}
    solid_numbers=list(final_dic.keys())# sorted solid based on high prob score
    score=list(final_dic.values())
    #print("Top",K,"Recommendations")
    TOP_K=[]
    for index in range(len(solid_numbers)):
        if solid_numbers[index] in SOL_IDs_Allocated_to_ER_ID:#check condition for solid allocated
            TOP_K.append(solid_numbers[index])   
        if len(TOP_K)==K:
            break
    return TOP_K
#############################################################################################################
def Solid_recommendations(model,K_recommends,ERName_Test,testing_data,SOL_ID_list_asper_RF_model,SOL_IDs_Allocated_to_ER_ID):
    #INPUT
    ERName_Test=ERName_Test[['ER ID']]# using er id and not er name
    ERName_Test_LIST=[]
    Recommendation_List=[]
    for each_user_index in tqdm(range(ERName_Test.shape[0])):
        user_sample=testing_data[each_user_index:each_user_index+1] #current user features
        Name=ERName_Test.iloc[each_user_index][0] # Current user UCIC value
        #customers features input
        # solid recommendations
        prediction_prob=model.predict_proba(user_sample)[0]
        tup_logo_score_dic={} #empty dic
        #all logo with prob
        for l_index in range(0,len(SOL_ID_list_asper_RF_model)):
            
            solid_number=SOL_ID_list_asper_RF_model[l_index]
            tup_logo_score_dic[str(solid_number)]=prediction_prob[l_index]

        #OUTPUT
        K=K_recommends #no of recomm
        TOP_K=Top_K_Rec(tup_logo_score_dic,K_recommends,SOL_IDs_Allocated_to_ER_ID) # Top K solid outputs
        #saving results for current Name
        ERName_Test_LIST.append(Name)
        Recommendation_List.append(TOP_K)
    return ERName_Test_LIST,Recommendation_List
##################################################################################    
    


# In[150]:


def Custom_Recommend_SOLID(model,K_recommends,Training_Data,ER_ID,solid_VRM_ACTIVE_df):  
   # Training_Data=Training_Data.rename(columns={'ER_ID':'ER ID','ER_NAME':'ER NAME'},in)# renaming columns from 'ER ID','ER NAME' to 'ER_ID','ER_NAME'
   # Training_Data.rename(columns = {'ER_ID':'ER ID'}, inplace = True)
   # Training_Data.rename(columns = {'ER_NAME':'ER NAME'}, inplace = True)
    ER_Names=Training_Data[['ER ID','ER NAME']] 
    
    Training_Data.drop(['SOLID','ER ID','ER NAME'],axis=1,inplace=True)
    Features=Training_Data
    ###########################################################
    Features['LASTDATEOFVISIT'] = Features['LASTDATEOFVISIT'].replace(['NOT_VISITED'], 9999)
    dates=Features['LASTDATEOFVISIT'].values
    today = datetime.date.today()
    for index in range(0,Features.shape[0]):
        if dates[index]!=9999:
            date_time_str = dates[index]
            date_time_obj = datetime.datetime.strptime(date_time_str, '%d %b %Y')
            date_time_obj = date_time_obj.date()
            diff=today-date_time_obj
            dates[index]=int(diff.days)
    dates_updated=[]
    for d_val in dates:
        if d_val<0:
            dates_updated.append(0)
        else:
            dates_updated.append(d_val)
    Features['LASTDATEOFVISIT']=dates_updated
    ############################################################
    Features.drop(['CITY','STATE','BRANCH','Bnkchl_Open_Date','LOCATION_DESCRIPTION'],axis=1,inplace=True)  #Bnkchl_Open_year
    #Features['LAST DATE OF VISIT']=Features['LAST DATE OF VISIT'].astype(int)
    objcolumnsName = Features.columns[Features.dtypes == 'object']
    Final_Data=pd.get_dummies(Features, columns=objcolumnsName)
    Final_Data.insert(loc=0, column='ER ID', value=ER_Names['ER ID'].values)
    Final_Data.insert(loc=1, column='ER Name', value=ER_Names['ER NAME'].values)
    Final_Data.drop_duplicates(subset=['ER ID','ER Name'], keep="last",inplace=True)
    Final_Data=Final_Data[Final_Data['ER ID']==int(ER_ID)]#selecting data for given ER_ID
    SOL_IDs_Allocated_to_ER_ID= list(solid_VRM_ACTIVE_df[solid_VRM_ACTIVE_df['ER ID']==str(ER_ID)]['SOL_ID'].values)
    #print(Final_Data.columns)
    ERName_Test=Final_Data[['ER ID','ER Name']]
   # print(ERName_Test,"abc")
    #print(Final_Data.columns)
    Final_Data=Final_Data.drop(['ER ID','ER Name','Bnkchl_Open_year'],axis=1) #added Bnkchl_Open_year on 13 Dec to counter feature mismatch from model. In original version this was there in the model and  deleted later.

    SOL_ID_list_asper_RF_model=model.classes_
    #print("***\n",Final_Data.columns,"***\n")
    #print(ERName_Test,"a\n",SOL_ID_list_asper_RF_model,"a\n",Final_Data)
    ERName_Test_LIST,Recommendation_List=Solid_recommendations(model,K_recommends,ERName_Test,Final_Data,\
                                                               SOL_ID_list_asper_RF_model,SOL_IDs_Allocated_to_ER_ID)
    #print("b",ERName_Test_LIST,Recommendation_List)
    return ERName_Test,Recommendation_List


# In[ ]:


model.feature_names


# In[151]:


#to_datetime year-date-month
def validation_checking(ER_ID,ER_name,rec_list,Original_DF):
    sample_df=Original_DF[(Original_DF['ER ID']==ER_ID) & (Original_DF['ER NAME']==ER_name)]
    
   # print(type(ER_ID),"\n",ER_ID,"\n")
    #sample_df['ER ID']=sample_df['ER ID'].astype(int)
    print(sample_df['SOLID'],"\n**",rec_list)
   # print(sample_df['ER ID'],"\n",type(sample_df['ER ID']))
    sample_df=sample_df[(sample_df['SOLID'].astype('str')).isin(rec_list)]
    #print(sample_df,"**\n")
    
    #sample_df=sample_df[['SOLID','ER ID','ER NAME', 'BRANCH', 'MEGA ZONE', 'Population_Category', 'LAST DATE OF VISIT','Latitude','Longitude','MODE OF VISIT', 'FOOTFALL', 'CONTROL NO', 'Branch_Vintage','Median_Branch_Vintage_Staff', 'Audit_rating', 'Attritions']]
    #sample_df.drop_duplicates(inplace=True)
    #sample_df['LASTDATEOFVISIT'] = pd.to_datetime(sample_df['LASTDATEOFVISIT'])
    sample_df=sample_df.sort_values('LASTDATEOFVISIT').drop_duplicates('SOLID',keep='last') #latest date
    #return sample_df
    not_found_vist=list(set(rec_list).difference(set(sample_df['SOLID'].values)))
    Not_Visited_DF=Original_DF[Original_DF['SOLID'].isin(not_found_vist)]#[['SOLID', 'BRANCH', 'MEGA ZONE', 'Population_Category','Latitude','Longitude','MODE OF VISIT', 'FOOTFALL', 'CONTROL NO', 'Branch_Vintage','Median_Branch_Vintage_Staff', 'Audit_rating', 'Attritions']]
    Not_Visited_DF.drop_duplicates(inplace=True)
    if 'ER ID' not in Not_Visited_DF.columns:
        Not_Visited_DF.insert(loc=1, column='ER ID', value=ER_ID)
    if 'ER NAME' not in Not_Visited_DF.columns:
        Not_Visited_DF.insert(loc=2, column='ER NAME', value=ER_name)
    if 'LASTDATEOFVISIT' not in Not_Visited_DF.columns:
        Not_Visited_DF.insert(loc=6, column='LASTDATEOFVISIT', value='NOT_VISITED') 
    Result_df=pd.concat((sample_df,Not_Visited_DF),axis=0,ignore_index=True)
    #print(Result_df)
    return Result_df


# In[152]:


#Loading
import datetime
from tqdm import tqdm
import pandas as pd
#Data=pd.read_csv('./python_data/Training_Data_with_lat_long_24-11-2022.csv',dtype={'ER ID':object,'SOLID':object})
#Data=Data[['ER ID', 'ER NAME', 'SOLID', 'CITY', 'STATE', 'BRANCH', 'MEGA ZONE','Population_Category', 'GL_SIZE_31-10-2022', 'LAST DATE OF VISIT','MODE OF VISIT', 'FOOTFALL', 'CONTROL NO', 'Branch_Vintage','Median_Branch_Vintage_Staff', 'Audit_rating', 'Attritions','BRANCH_SIZE']]

VRM_DATA_PATH='./python_data/VRM_DATA/'
data_path_VRM_ACTIVE=VRM_DATA_PATH+'VRM_ACTIVE_1_data.xlsx'
solid_VRM_ACTIVE_df=pd.read_excel(data_path_VRM_ACTIVE, dtype={'SOL_ID': 'str','ER ID': 'str'})
solid_VRM_ACTIVE_df=solid_VRM_ACTIVE_df[['SOL_ID','ER ID','ER NAME']]
solid_VRM_ACTIVE_df.drop_duplicates(inplace=True)
solid_VRM_ACTIVE_df.dropna(inplace=True)


# In[ ]:


zz=pd.read_excel(data_path_VRM_ACTIVE)
zz


# In[159]:


#Loading
#Train_DF=pd.read_csv(VRM_DATA_PATH_JULY2023+'Training_Data_with_lat_long_25-07-2023.csv',dtype={'ER ID':object,'SOLID':object})


# In[21]:


Train_DF1=Train_DF #First
#Train_DF=Train_DF1.copy() #second


# In[ ]:


df=pd.DataFrame


# In[ ]:





# In[153]:


#for i in range()
#Train_DF['SOLID']=Train_DF['SOLID'].zfill()#
Train_DF['LASTDATEOFVISIT']


# In[160]:


Train_DF.rename(columns = {'ER_ID':'ER ID'}, inplace = True)
Train_DF.rename(columns = {'ER_NAME':'ER NAME'}, inplace = True)


# In[161]:


Train_DF.rename(columns = {'City':'CITY'}, inplace = True)
Train_DF.rename(columns = {'State':'STATE'}, inplace = True)


# In[156]:


Features.columns


# In[ ]:





# # Json Dump

# In[157]:


ERName_Test_LIST,Recommendation_List=Custom_Recommend_SOLID(model,Top_K,Train_DF,ER_ID,solid_VRM_ACTIVE_df)


# In[ ]:


Recommendation_List


# In[ ]:


ERName_Test_LIST


# In[ ]:


ERName_Test_LIST


# In[ ]:


Train_DF.columns


# In[15]:


Training_Data.rename(columns = {'ER_ID':'ER ID'}, inplace = True)
Training_Data.rename(columns = {'ER_NAME':'ER NAME'}, inplace = True)


# In[163]:


Train_DF.columns


# In[16]:


#Train_DF1.to_csv(VRM_DATA_PATH_JULY2023+'Train_DF1.csv')
Train_DF=pd.read_csv(VRM_DATA_PATH_JULY2023+'Train_DF1.csv')
Train_DF=Train_DF.drop('Unnamed: 0',axis=1)


# In[62]:


Train_DF.shape


# In[66]:


#Train_DF['ER ID'].nunique()


# In[164]:


#INPUT############
from tqdm import tqdm
counter=0
for erid in tqdm(set(Train_DF['ER ID'].values)):
    Top_K=5
    col_list=["Top_"+str(i) for i in range(0,Top_K)]
    ER_ID=int(erid)
    Data_DF=Train_DF.copy()
    ERName_Test_LIST,Recommendation_List=Custom_Recommend_SOLID(model,Top_K,Data_DF,ER_ID,solid_VRM_ACTIVE_df)### Recommendations
    ############################################
    #print(ERName_Test_LIST['ER ID'].values,"\n",ERName_Test_LIST['ER Name'].values,"\n",Recommendation_List)
    #print(Recommendation_List)
    if len(Recommendation_List)==0:
        continue
    #OUTPUT######
    try:
        Recommendation_df2 = pd.DataFrame.from_records(Recommendation_List,columns=col_list)
    except:
        difference=len(col_list)-len(Recommendation_List[0])
        for i in range(difference):
            Recommendation_List[0].append('Null')
        Recommendation_df2 = pd.DataFrame.from_records(Recommendation_List,columns=col_list)    
    Recommendation_df2.insert(loc=0, column='ER ID', value=ERName_Test_LIST['ER ID'].values)
    Recommendation_df2.insert(loc=1, column='ER Names', value=ERName_Test_LIST['ER Name'].values)
    #print(Recommendation_df2)
    #Recommendation_df=pd.concat([Recommendation_df,Recommendation_df2],axis=0)
    ER_ID=Recommendation_df2['ER ID'].iloc[0]
    ER_name=Recommendation_df2['ER Names'].iloc[0]
    #print(ER_ID,"\n",ER_name)
    rec_list=Recommendation_df2.iloc[0].values[2:]
    rec_list=[str(i) for i in rec_list]
    #print(rec_list)
    #print(Recommendation_df.iloc[index])
    if counter==0:
        
        Result_df=validation_checking(ER_ID,ER_name,rec_list,Train_DF)
        #for col in ['BRANCH','LAST DATE OF VISIT']:
            #if col in Result_df.columns:
                #Result_df.drop_duplicates([col],inplace=True)
        counter=1
    else:
        Result_df2=validation_checking(ER_ID,ER_name,rec_list,Train_DF)
        #for col in ['BRANCH','LAST DATE OF VISIT']:
            #if col in Result_df2.columns:
                #Result_df2.drop_duplicates([col],inplace=True)
        #print('abc')
        Result_df=pd.concat([Result_df,Result_df2],axis=0)
    


# In[20]:


#Experimental: take highest frequency region from the recommendations
new_df=Result_df.groupby('ER ID')['Region'].agg(lambda x: (x.mode().iloc[0] if not x.mode().empty else None,x.count())).reset_index()
#new_df.columns=['ER ID','moste_frequent_region','frequency']


# In[21]:


assigned_region=pd.DataFrame()


# In[22]:


assigned_region['ER ID']=new_df['ER ID']
assigned_region['Region']=new_df['Region']
assigned_region['region_final']='abc'


# In[23]:


for i in range(len(assigned_region)):
    assigned_region['region_final'][i]=assigned_region['Region'][i][0]


# In[24]:


assigned_region


# In[36]:


assigned_Region1=assigned_region[['ER ID','region_final']]


# In[37]:


# code to pick recommendations only from region_final
merged_df=pd.merge(Result_df,assigned_Region1,on='ER ID')
final_result_df=merged_df[merged_df['Region']==merged_df['region_final']]


# In[27]:


final_result_df=final_result_df.drop('region_final',axis=1)


# In[38]:


Result_df=final_result_df.copy()


# In[39]:


Result_df.reset_index(inplace=True)


# In[40]:


Result_df.drop(['index'],axis=1,inplace=True)


# In[ ]:





# In[41]:


Result_df.drop(['Bnkchl_Open_year'],axis=1,inplace=True)


# In[42]:


Result_df1=Result_df[['SOLID', 'ER ID', 'ER NAME','region_final']]


# In[43]:


# recommendations
Result_df1


# In[45]:


Result_df1.to_excel('Single Region Recommendations.xlsx')


# In[199]:


Result_df1['ER ID']=Result_df1['ER ID'].astype(str)


# In[ ]:


#Result_df1['ER ID']=[i.replace('.0','') for i in Result_df1['ER ID']]


# In[201]:


Original_Data=pd.read_excel(VRM_DATA_PATH_JULY2023+'ERSystemAPI_Data_CompleteFeats_25-07-2023.xlsx',dtype={'SOLID':object,'ER_ID':object})


# In[ ]:


Original_Data


# In[ ]:


Original_Data.drop(['ER_ID','ER_NAME'],axis=1,inplace=True)


# In[202]:


Original_Data.head()


# In[203]:


Original_Data.drop_duplicates(inplace=True)


# In[205]:


Final_Recommendation=pd.merge(Result_df1,Original_Data,on='SOLID',how='left')


# In[ ]:


Result_df1['SOLID']=Result_df1['SOLID'].astype('str')


# In[ ]:


Final_Recommendation


# In[ ]:


Add_Columns_DF.rename({'solId':'SOLID'},inplace=True,axis=1)
Add_Columns_DF.columns


# In[ ]:


Recommend_Result_DF = pd.merge(Final_Recommendation,Add_Columns_DF, on=['SOLID'], how='left')
Recommend_Result_DF.head()


# In[ ]:


#duplicates were there at this stage in previous instances also , were later removed, bringing that code here itself.
Recommend_Result_DF=Recommend_Result_DF.drop_duplicates(['SOLID','ER ID'],keep= 'last')


# In[205]:


Train_DF=pd.read_csv(VRM_DATA_PATH_JULY2023+'Training_Data_with_lat_long_25-07-2023.csv',dtype={'ER ID':object,'SOLID':object})


# In[206]:


Train_DF.head()


# In[ ]:


Train_DF=Train_DF[['SOLID','Branch_Vintage']]
Train_DF['SOLID']=Train_DF["SOLID"].astype('str')


# In[ ]:


Recommend_Result_DF_Final = pd.merge(Recommend_Result_DF,Train_DF, on=['SOLID'], how='left')
Recommend_Result_DF_Final.head()


# In[ ]:


#saving
Recommend_Result_DF_Final.to_csv(VRM_DATA_PATH_JULY2023+'Recommend_Result_DF_25_07_2023.csv',index=False)


# In[209]:


#loading
import pandas as pd 
Recommend_Result_DF_Final=pd.read_csv(VRM_DATA_PATH_JULY2023+'Recommend_Result_DF_25_07_2023.csv',dtype={'SOLID':object,"ER ID":object})


# In[ ]:


Recommend_Result_DF_Final.drop_duplicates(inplace=True)


# In[ ]:


Recommend_Result_DF_Final['LASTDATEOFVISIT'].fillna("NULL",inplace=True)


# In[ ]:


Recommend_Result_DF_Final.LASTDATEOFVISIT[Recommend_Result_DF_Final.LASTDATEOFVISIT == 'NULL'] = 'NOT_VISITED'


# In[ ]:


Recommend_Result_DF_Final['LASTDATEOFVISIT'].value_counts(dropna=False)


# In[ ]:


Recommend_Result_DF_Final


# In[ ]:


Recommend_Result_DF=Recommend_Result_DF_Final.sort_values('LASTDATEOFVISIT')
Recommend_Result_DF


# In[ ]:


Recommend_Result_DF=Recommend_Result_DF.drop_duplicates(['SOLID','ER ID'],keep= 'last')


# In[ ]:





# In[ ]:


#Regions are not present in final recommendation for some reason, bringing them here from Train_DF
Train_DF['SOLID']=Train_DF['SOLID'].astype('str')
Recommend_Result_DF_reg=pd.merge(Recommend_Result_DF,Train_DF[['SOLID','Region']],on='SOLID',how='left')


# In[207]:


Recommend_Result_DF_Apr2023=pd.read_csv(VRM_DATA_PATH+'APR2023/Recommend_Result_DF_20_04_2023.csv',dtype=str)


# In[ ]:


Recommend_Result_DF_Apr2023_SOLID_list=list(set(Recommend_Result_DF_Apr2023['SOLID'].to_list()))
len(Recommend_Result_DF_Apr2023_SOLID_list)


# In[ ]:


Recommend_Result_DF = Recommend_Result_DF[~Recommend_Result_DF['SOLID'].isin(Recommend_Result_DF_Apr2023_SOLID_list)]


# In[ ]:


Recommend_Result_DF.shape


# In[206]:


Recommend_Result_DF.head()


# In[ ]:


Recommend_Result_DF_reg.drop_duplicates()


# In[ ]:


Recommend_Result_DF_reg['LASTDATEOFVISIT'][4][0]


# In[ ]:


Recommend_Result_DF_reg['LASTDATEOFVISIT'][4][-8:]


# In[ ]:


for i in range(len(Recommend_Result_DF_reg)):
    if (Recommend_Result_DF_reg['LASTDATEOFVISIT'][i][-8:]=="Jan 2023"):#and Recommend_Result_DF_reg['LASTDATEOFVISIT'][i][-8:-5]=='Jan'):
        print('a')


# In[ ]:


custom_reg_df=Recommend_Result_DF_reg.loc[Recommend_Result_DF_reg['Region'].isin(regions_Punjab)]
                                                                                


# In[ ]:


custom_reg_df=custom_reg_df.drop_duplicates()


# # Routing Distance

# In[ ]:





# # Final JSON DUMP for API

# In[ ]:


LIST= [Recommend_Result_DF.iloc[index].to_dict() for index in range(Recommend_Result_DF.shape[0])]


# In[ ]:





# # final integration of shortest distance algorithm

# In[ ]:


LIST= [Recommend_Result_DF.iloc[index].to_dict() for index in range(Recommend_Result_DF.shape[0])]


# In[ ]:


LIST


# In[ ]:


import json
path_json=VRM_DATA_PATH_JULY2023+'Recommendation_df_Original_25-07-2023.json'


# In[ ]:


type(LIST)


# In[ ]:


import json
with open(path_json, 'w') as f:
    f.write(str(LIST))


# # Shortest Path

# In[ ]:


recommendation={}
x=Recommend_Result_DF['ER ID'].unique()
x
for i in x:
    recommendation[i]=[]


# In[ ]:


recommendation={}
x=Recommend_Result_DF['ER ID'].unique()
for i in x:
    recommendation[i]=Recommend_Result_DF[Recommend_Result_DF['ER ID']==i]['SOLID'].to_list()


# In[ ]:


#extract ER ID

  #recommendation[Recommend_Result_DF['ER ID'][0]]=[]
    
# this logic will break in case of single suggestion being at the last.
  
for i in range(0,len(Recommend_Result_DF)):
      print(i)
      if i==0:
        if  Recommend_Result_DF['ER ID'][i]!=Recommend_Result_DF['ER ID'][i+1]:
            recommendation[Recommend_Result_DF['ER ID'][i]].append(Recommend_Result_DF['SOLID'][i])
        else:
            recommendation[Recommend_Result_DF['ER ID'][i]].append(Recommend_Result_DF['SOLID'][i])
    
      else:
        if i !=len(Recommend_Result_DF)-1 :
            if Recommend_Result_DF['ER ID'][i]==Recommend_Result_DF['ER ID'][i-1]:
                recommendation[Recommend_Result_DF['ER ID'][i]].append(Recommend_Result_DF['SOLID'][i])
            elif Recommend_Result_DF['ER ID'][i]!=Recommend_Result_DF['ER ID'][i-1] and Recommend_Result_DF['ER ID'][i]==Recommend_Result_DF['ER ID'][i+1] :
                recommendation[Recommend_Result_DF['ER ID'][i]].append(Recommend_Result_DF['SOLID'][i])
            elif Recommend_Result_DF['ER ID'][i]!=Recommend_Result_DF['ER ID'][i-1] and Recommend_Result_DF['ER ID'][i]!=Recommend_Result_DF['ER ID'][i+1]:
                recommendation[Recommend_Result_DF['ER ID'][i]].append(Recommend_Result_DF['SOLID'][i])
        else:
            if Recommend_Result_DF['ER ID'][i]!=Recommend_Result_DF['ER ID'][i-1]:
                recommendation[Recommend_Result_DF['ER ID'][i]].append(Recommend_Result_DF['SOLID'][i])

  


# In[ ]:


import pandas as pd
import geopy as gp
from geopy import distance
from math import sin, cos, sqrt, atan2, radians
import folium
from IPython.display import display


# In[ ]:


x=pd.read_csv('0911_Branch_Lat_long.csv')
x.head()


# In[ ]:


def read_lat_long(filename):
    x=pd.read_csv(filename,encoding='ISO-8859-1')
    return x


# In[ ]:


def find_lat_long_from_sol_id(solid):
 x=read_lat_long('0911_Branch_Lat_long.csv')
 for i in range(len(x)):
    if x['Sol ID'][i]==int(solid):
        return (x['Latitude'][i]),(x['Longitude'][i])
def find_branch_from_sol_id(solid):
    x=read_lat_long('0911_Branch_Lat_long.csv')
    for i in range(len(x)):
        if x['Sol ID'][i]==int(solid):
            return x['Branch-Name'][i]


# In[ ]:


def get_distance(point1, point2):
    R = 6370
    lat1 = radians(point1[0])  #insert value
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2- lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance


# In[ ]:


def dist(priority):
    dictionary={}
    distance=[]
    for i in priority:
        for j in priority:
            if i==j:
                dictionary[j]=0
            else:
                (lat1,long1)=find_lat_long_from_sol_id(i)
                (lat2,long2)=find_lat_long_from_sol_id(j)
                dist=get_distance((lat1,long1),(lat2,long2))
                dictionary[j]=dist
        distance.append(dictionary)
        dictionary={}
    return distance


# In[ ]:


import networkx as nx
import numpy as np

def distance_adjency_matrix(branch_list):
    distance_branches=[]
    for i in branch_list:
        temp=[]
        for j in branch_list:
            if i!=j:
                (lat1,long1)=find_lat_long_from_sol_id(i)
                (lat2,long2)=find_lat_long_from_sol_id(j)
                dist=get_distance((lat1,long1),(lat2,long2))
                temp.append(dist)
               
            else:
                temp.append(0)
        distance_branches.append(temp)
    return distance_branches
        
        
def shortest_path(source_branch,priority_order):
    
    branch_list=priority_order
    #branch_list.append(source_branch)
    distance_branches=distance_adjency_matrix(branch_list)
    print(distance_branches)
    #matrix_np = np.matrix(distance_branches)
    adjency_DF=pd.DataFrame(distance_branches,index=priority_order,columns=priority_order)
    original_graph = nx.from_pandas_adjacency(adjency_DF)
    mst=nx.minimum_spanning_tree(original_graph,algorithm='kruskal')
    return mst,original_graph,adjency_DF
                
    
        
    


# In[ ]:


priority2=list(Recommend_Result_DF[Recommend_Result_DF['ER ID']=='294623']['SOLID'].to_numpy())
source_branch=priority2[0]
mst,G,adjency_DF=shortest_path(source_branch,priority2)


# In[ ]:


nx.draw_networkx(nx.from_pandas_adjacency(adjency_DF))


# In[ ]:


pos = nx.circular_layout(mst)  
nx.draw_networkx(mst,pos=pos) 
nx.draw_networkx_edge_labels(mst,pos,edge_labels=mst.edges)


# In[ ]:


Result_df=Recommend_Result_DF.copy()


# In[ ]:


#extraction of GL size,attrition,population category, phone, branch manager

#def extract_gl(solid):

def extract_branch_manager(solid):
    x=pd.read_excel('SOL_BRANCH_DETAILS (1)')
    
    
def extract_attrition(solid):
    for i in range(len(Result_df)):
        if Result_df["SOLID"][i]==solid:
            return Result_df['Attritions'][i]


def extract_population_category(solid):
    for i in range(len(Result_df)):
        if Result_df["SOLID"][i]==solid:
            return Result_df['Population_Category'][i]
        
def extract_glsize(solid):
    for i in range(len(Result_df)):
        if Result_df["SOLID"][i]==solid:
            return Result_df['GL_SIZE'][i] 
        
def extract_branchmangername(solid):
    for i in range(len(Result_df)):
        if Result_df["SOLID"][i]==solid:
            return Result_df['branchManager.displayName'][i]  
        
def extract_branchphone(solid):
    for i in range(len(Result_df)):
        if Result_df["SOLID"][i]==solid:
            return Result_df['branchLandlineNumber'][i]  


# In[ ]:


for i in list(mst.edges):
    lat1,long1=find_lat_long_from_sol_id(i[0])
    lat2,long2=find_lat_long_from_sol_id(i[1])
    folium.PolyLine(locations = [(lat1,long1),(lat2,long2)],
                line_opacity = 0.5).add_to(mymap)
    


# In[ ]:


#single function call to perform everything.
def combined(priority_order):
    # part one : find distances
    x=dist(priority_order)
    source_branch=priority_order[0]
    mst,G,adjency_DF=shortest_path(source_branch,priority_order)
    #draw graph
    nx.draw_networkx(nx.from_pandas_adjacency(adjency_DF))
    #draw mst
    pos = nx.circular_layout(mst)  
    nx.draw_networkx(mst,pos=pos) 
    nx.draw_networkx_edge_labels(mst,pos,edge_labels=mst.edges)
    #draw single routes
    priority_string=['very high','high','medium','low','very_low']
    mymap=markonmap(priority2)
    #print may not work
    print(mymap)
    for i in list(mst.edges):
        lat1,long1=find_lat_long_from_sol_id(i[0])
        lat2,long2=find_lat_long_from_sol_id(i[1])
        folium.PolyLine(locations = [(lat1,long1),(lat2,long2)],
                    line_opacity = 0.5).add_to(mymap)
    #print may not work
    print(mymap)
    #draw multi route(FC graph of branches)
    priority_string=['very high','high','medium','low','very_low']
    mymap2=markonmap(priority2)
    for i in list(G.edges):
        lat1,long1=find_lat_long_from_sol_id(i[0])
        lat2,long2=find_lat_long_from_sol_id(i[1])
        folium.PolyLine(locations = [(lat1,long1),(lat2,long2)],line_opacity = 0.5).add_to(mymap2)
    print(mymap2)
        
    return x,mst,G,adjency_df,mymap,mymap2



    
    
    
    
    


# In[ ]:


#add distance to output df
from tqdm import tqdm
distances={}
for i in tqdm(recommendation):
    #for j in recommendations[i]:
       
        x=dist(recommendation[i])
        if x:
            distances[i]=x[0]
        
          


# In[ ]:


#improvised for this use case 
import networkx as nx
import numpy as np
from networkx.algorithms import tree

def distance_adjency_matrix(branch_list):
    distance_branches=[]
    for i in branch_list:
        temp=[]
        for j in branch_list:
            if i!=j:
                (lat1,long1)=find_lat_long_from_sol_id(i)
                (lat2,long2)=find_lat_long_from_sol_id(j)
                dist=get_distance((lat1,long1),(lat2,long2))
                temp.append(dist)
               
            else:
                temp.append(0)
        distance_branches.append(temp)
    return distance_branches
        
        
def shortest_path(source_branch,priority_order):
    
    branch_list=priority_order
    #branch_list.append(source_branch)
    distance_branches=distance_adjency_matrix(branch_list)
    #print(distance_branches)
    #matrix_np = np.matrix(distance_branches)
    adjency_DF=pd.DataFrame(distance_branches,index=priority_order,columns=priority_order)
    original_graph = nx.from_pandas_adjacency(adjency_DF)
    mst=tree.minimum_spanning_edges(original_graph, algorithm="kruskal", data=False)
    return mst,original_graph,adjency_DF
                
    
        
    


# In[ ]:


result={}
for i in tqdm(distances):
    mst,original_graph,adjency_DF=shortest_path(list(distances[i].keys())[0],distances[i].keys())
   # temp=[]
    
    #temp.append(mst)
    #temp.append(original_graph)
    
    #temp.append(adjency_DF)
    result[i]=mst
    


# In[ ]:





# In[ ]:


result


# In[ ]:


visit_order={}
for i in result:
    #print(result[i])
    edgelist = list(result[i])
    x=sorted(sorted(e) for e in edgelist)
    visit_order[i]=x


# In[ ]:


visit_order['414951']


# In[ ]:


Recommend_Result_DF1=Recommend_Result_DF


# In[ ]:





# In[ ]:


def merge(visit_order):
    q=[]
    for i in visit_order:
        if i[0]  not in q:
            q.append(i[0])
        if i[1]  not in q:
            q.append(i[1])
    return q


# In[ ]:


merged_visit_order={}
for i in visit_order:
    x=merge(visit_order[i])
    merged_visit_order[i]=x


# In[ ]:


merged_visit_order


# In[ ]:


Recommend_Result_DF1['visit_order']=0


# In[ ]:


Recommend_Result_DF1 = Recommend_Result_DF1.reset_index(drop=True)


# In[ ]:


Recommend_Result_DF1


# In[ ]:


for i in range(len(Recommend_Result_DF1)):
   for j in merged_visit_order:
       print(i,j)
       if Recommend_Result_DF1['ER ID'][i]==j:
            for k in merged_visit_order[j]:
                if Recommend_Result_DF1['SOLID'][i]==k:
                  Recommend_Result_DF1['visit_order'][i]=(merged_visit_order[j].index(k))+1
        

        
        #Recommend_Result_DF1['path'][i]=visit_order[j]
      


# In[ ]:


Recommend_Result_DF1.head(11)


# In[ ]:


for i in  range(0,len(Recommend_Result_DF1)):
    if Recommend_Result_DF1['visit_order'][i]==0:
        Recommend_Result_DF1['visit_order'][i]=1
        
    


# In[ ]:


#saving
Recommend_Result_DF1.to_csv(VRM_DATA_PATH_APR2023+'Recommend_Result_DF1_20_04_2023.csv',index=False)


# In[ ]:


#loading
import pandas as pd 
Recommend_Result_DF1=pd.read_csv(VRM_DATA_PATH_APR2023+'Recommend_Result_DF1_20_04_2023.csv',dtype={'SOLID':object,'ER ID':object})


# In[ ]:


Recommend_Result_DF1


# In[ ]:


# Last March Month 288 recommendations were sent will start from 289 onwards
Rec_number=[]
for i in range(289,289+Recommend_Result_DF1.shape[0]):
    Rec_number.append(i)
print(len(Rec_number))
    
Recommend_Result_DF1.insert(loc=0, column='Recommend_No', value=Rec_number)


# In[ ]:


Recommend_Result_DF1


# In[ ]:


Recommend_Result_DF1['SOLID']=Recommend_Result_DF1['SOLID'].astype(str)
Recommend_Result_DF1['ER ID']=Recommend_Result_DF1['ER ID'].astype(str)


# In[ ]:


Recommend_Result_DF1.to_json(orient="records")


# In[ ]:


# storing the data in JSON format
Recommend_Result_DF1.to_json('./python_data/Recommendations_Output/Recommend_Result_DF1_20_04_2023_purejson.json', orient = 'records')
 
# reading the JSON file
Recommend_Result_DF1 = pd.read_json('./python_data/Recommendations_Output/Recommend_Result_DF1_20_04_2023_purejson.json', orient ='records')

