#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np


# In[11]:


#Series
labels = ['a', 'b', 'c','d','e']
my_data = [10,20,30,40,50]
arr = np.array(my_data)
d = {'a':10, 'b':20, 'c':30}


# In[9]:


pd.Series(data = my_data)


# In[18]:


pd.Series(data=my_data, index=labels)


# In[19]:


pd.Series(arr)


# In[21]:


pd.Series(arr, labels)


# In[22]:


pd.Series(d) 


# In[23]:


pd.Series(labels)


# In[24]:


pd.Series(data=[sum,print,len])


# In[27]:


ser1 = pd.Series([1,2,3,4],['IND','USA','AUS','JAPAN'])
ser1


# In[41]:


ser2 = pd.Series([1,2,5,6],['IND', 'Nepal', 'SriLanka', 'Maldives'])
ser2


# In[29]:


ser1['IND']


# In[31]:


ser3 = pd.Series(data=labels)
ser3


# In[35]:


ser3[3]


# In[36]:


ser1


# In[42]:


ser2


# In[43]:


ser1+ser2


# In[44]:


#DATAFRAMES


# In[63]:


from numpy.random import randn


# In[64]:


np.random.seed(101)


# In[54]:


df = pd.DataFrame(randn(5,4), ['A','B','C','D','E'],['W','X','Y','Z'])
df


# In[13]:


df['W']


# In[14]:


type(df['W'])


# In[15]:


type(df)


# In[19]:


df.W         #dont use this, as it may confuse panda with the existing methods


# In[18]:


df.W['D']


# In[55]:


#getting multiple columns-
df[['W','X']]


# In[44]:


#creating a new columns-
df['NEW'] = df['W'] + df['X']


# In[23]:


df


# In[35]:


df.drop('NEW', axis=1)


# In[36]:


df


# In[45]:


# to delete the data from panda use 'inplace' as true
df.drop('NEW', axis=1, inplace = True)


# In[46]:


df


# In[47]:


df.drop('E')


# In[48]:


df


# In[50]:


df.drop('E', inplace = True)


# In[51]:


df


# In[52]:


df.shape


# In[57]:


df


# In[56]:


df.shape


# In[63]:


#seelct items in column
df['W']


# In[65]:


#select items in row
df.loc['E']


# In[66]:


df.loc['E']['Y']


# In[68]:


df.iloc[2]


# In[69]:


#selecting subset
df.loc['E','Y']


# In[70]:


df.loc['B','Y']


# In[71]:


df


# In[72]:


df.loc[['A','B'],['W','Y']]


# In[73]:


#conditional selection
df > 0


# In[74]:


boolDf = df>0


# In[75]:


boolDf


# In[76]:


df[boolDf]


# In[77]:


df[df<0]


# In[78]:


df[df>0]


# In[79]:


df['W']>0


# In[80]:


df[df['W']>0]


# In[81]:


df


# In[82]:


bd = df[df['W']>0]


# In[83]:


bd['W']


# In[88]:


#get all dataframe where Z is greater than 0
df['Z']>0


# In[89]:


df[df['Z']>0]


# In[87]:


md = df[df['Z']>0]
md['Z']


# In[90]:


df[df['Z']>0].Z


# In[91]:


df[df['Z']>0]['Z']


# In[92]:


df[df['Z']>0][['X','Z']]


# In[98]:


df[df['Z']>0].loc['B']


# In[105]:


df[df['Z']>0].loc['B','Z']


# In[106]:


df


# In[108]:


df[(df['Y']>0) and (df['W']>1)]


# In[115]:


df[(df['Y']>0) | (df['W']>1)]


# In[116]:


df[(df['Y']>0) & (df['W']>1)]


# In[117]:


df


# In[119]:


df.reset_index()


# In[120]:


newInd = 'CA OR WY NY CO'.split()
newInd


# In[121]:


df['States'] = newInd
df


# In[122]:


df.set_index('States')


# In[123]:


df


# In[125]:


df.set_index('States', inplace= True)


# In[126]:


df


# In[14]:


#multi index and index heirarchy
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_indexs = pd.MultiIndex.from_tuples(hier_index)


# In[6]:


outside


# In[7]:


inside


# In[8]:


list(zip(outside,inside))


# In[9]:


pd.MultiIndex.from_tuples(hier_index)


# In[15]:


hier_indexs


# In[31]:


df = pd.DataFrame(randn(6,2), hier_indexs, ['A','B'])


# In[32]:


df


# In[33]:


df.loc['G1']


# In[35]:


df.loc['G1'].loc[1]


# In[37]:


df.loc['G1']['A']


# In[41]:


#naming index
df.index.names


# In[43]:


df.index.names = ['Groups', 'Num']
df


# In[44]:


df.loc['G2'].loc[2]['B']


# In[46]:


df.loc['G1'].loc[2]['A']


# In[48]:


df.loc['G2'].loc[3]['A']


# In[49]:


#cross section
df.loc['G1']


# In[50]:


df.xs('G1')


# In[51]:


df


# In[53]:


df.xs(1, level = 'Num')


# In[61]:


df.xs(3, level = 'Num')


# In[77]:


#Missing Date
import pandas as pd
import numpy as np


# In[82]:


d = {'A' : [1,2, np.nan], 'B' : [5,np.nan, np.nan], 'C' : [1,2,3]}


# In[84]:


df = pd.DataFrame(d)
df


# In[85]:


df.dropna()#row drop - use inplace = true to drop permanently


# In[87]:


df.dropna(axis=1)#column drop - use inplace = true to drop permanently 


# In[89]:


df


# In[94]:


df.dropna(thresh = 2)


# In[96]:


#filling missing value
df.fillna(value= 'Fill Value')


# In[98]:


#filling mean of value
df['A']


# In[101]:


df['A'].fillna(value=df['A'].mean())


# In[103]:


df.isnull()


# In[104]:


df.notnull()


# In[105]:


df.isna()


# In[106]:


df


# In[122]:


df['D'] = df['A']+df['B']


# In[124]:


df


# In[131]:


df.drop('D', axis=1)


# In[128]:


df.dropna(thresh=3)


# In[132]:


df


# In[134]:


df.drop('D', axis=1, inplace=True)


# In[135]:


df


# In[136]:


df.fillna('New')


# In[137]:


df


# In[139]:


df['A'].fillna(df['A'].mean())


# In[140]:


#Groupby


# In[142]:


import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}


# In[143]:


data


# In[146]:


df = pd.DataFrame(data)
df


# In[148]:


byComapny = df.groupby('Company')


# In[149]:


byComapny


# In[150]:


byComapny.mean()


# In[151]:


byComapny.max()


# In[152]:


byComapny.min()


# In[155]:


df


# In[156]:


byComapny.sum()


# In[157]:


byComapny.std()


# In[160]:


byComapny.sum().loc['GOOG']


# In[163]:


df.groupby('Company').sum().loc['FB']


# In[165]:


df.groupby('Company').count()


# In[166]:


df.describe()


# In[168]:


df.groupby('Company').describe()


# In[175]:


df.groupby('Company').describe().transpose()


# In[188]:


df.groupby('Company').describe().transpose()['FB']


# In[184]:


df.groupby('Company').describe().loc['FB']


# In[189]:


#Merging Joining and Concatenating


# In[191]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])


# In[192]:


df1


# In[193]:


df2


# In[194]:


df3


# In[196]:


pd.concat([df1,df2,df3])


# In[197]:


pd.concat([df1,df2,df3], axis=1)


# In[198]:


left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})   


# In[199]:


left


# In[200]:


right


# In[201]:


pd.concat([left, right])


# In[202]:


pd.concat([left, right], axis=1)


# In[204]:


pd.merge(left, right,'inner', on = 'key')


# In[205]:


pd.merge(left, right, 'outer', on='key')


# In[206]:


left1 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right1 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})


# In[207]:


left1


# In[208]:


right1


# In[210]:


pd.merge(left1, right1, how='inner', on=['key1','key2'])


# In[212]:


pd.merge(left1, right1, how='outer', on = ['key1','key2'])


# In[213]:


pd.merge(left1, right1, how='right', on = ['key1','key2'])


# In[214]:


pd.merge(left1, right1, how='left', on = ['key1','key2'])


# In[215]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


# In[217]:


left.join(right)


# In[218]:


right.join(left)


# In[219]:


left.join(right, how='outer')


# In[221]:


#Operations
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})


# In[222]:


df


# In[223]:


df.head()


# In[226]:


#finding unique value in dataframe
df['col2'].unique()


# In[230]:


len(df['col2'].unique())


# In[232]:


df['col2'].nunique()


# In[233]:


df['col2'].value_counts()


# In[236]:


df['col2'].unique()


# In[237]:


#selecting data
df


# In[238]:


df['col1']>2


# In[239]:


df[df['col1']>2]


# In[240]:


df[(df['col1']>2) & (df['col2'] > 500)]


# In[243]:


#Apply method
#apply custom function

def times2(x):
    return x**2


# In[244]:


df


# In[242]:


df['col2'].sum()


# In[245]:


df['col2'].apply(times2)


# In[246]:


def lenString(x):
    return len(x)


# In[247]:


df['col3'].apply(lenString)


# In[250]:


#applying with lambda function
df['col2'].apply(lambda x : x*2)


# In[257]:


df.drop('col3', axis=1)


# In[258]:


df


# In[259]:


df.columns


# In[263]:


type(df['col3'].loc[1])


# In[264]:


df.index


# In[266]:


#ordering/sorting 
df.sort_values(by='col2')


# In[267]:


df.isnull()


# In[268]:


data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}
df = pd.DataFrame(data)


# In[269]:


df


# In[270]:


#pivot table - creating mutiple index for given table
df.pivot('C',['A','B'])


# In[277]:


df.pivot_table(values='D', index=['A','B'], columns='C')


# In[303]:


#Data Input and Output
#CSV, Excel, Html, Sql
import pandas as pd


# In[293]:


pd.read_csv("/Users/divya/Udemy_sampleFiles/example")


# In[295]:


df = pd.read_csv("/Users/divya/Udemy_sampleFiles/example")
df


# In[297]:


df.to_csv("/Users/divya/Udemy_sampleFiles/myoutput_sample.csv")


# In[299]:


df.to_csv("/Users/divya/Udemy_sampleFiles/myUoutput_example_withoutIndex.csv", index=False)


# In[300]:


df


# In[301]:


pd.read_csv("/Users/divya/Udemy_sampleFiles/myoutput_sample.csv")


# In[304]:


pd.read_csv("/Users/divya/Udemy_sampleFiles/myUoutput_example_withoutIndex.csv")


# In[318]:


pd.read_excel("/Users/divya/Udemy_sampleFiles/Excel_Sample.xlsx", sheet_name='Sheet1')


# In[319]:


df.to_excel("/Users/divya/Udemy_sampleFiles/Excel_Sample_Output.xlsx",sheet_name='NewSHEET', index=False)


# In[321]:


pd.read_excel("/Users/divya/Udemy_sampleFiles/Excel_Sample_Output.xlsx",sheet_name='NewSHEET')


# In[322]:


#reading html
data = pd.read_html("https://fdic.gov/bank-failures/failed-bank-list")


# In[323]:


data


# In[325]:


df = pd.DataFrame(data[0])


# In[326]:


df


# In[347]:


fdic =pd.DataFrame(pd.read_csv('/Users/divya/Udemy_sampleFiles/bank_fdic.csv'))
fdic


# In[336]:


df


# In[337]:


df.head()


# In[338]:


data[0].head()


# In[ ]:


#reading from sql


# In[345]:


from sqlalchemy import create_engine

engine = create_engine('sqlite:///:memory:')

df.to_sql('my_table', engine)


# In[346]:


pd.read_sql('my_table', engine)


# In[353]:


fdic.to_sql('my_newTable2', engine, index=False)


# In[357]:


pd.read_sql('my_newTable', engine)


# In[352]:


pd.read_sql('my_newTable2', engine)


# In[358]:


pd.read_sql('my_table', engine)


# In[ ]:




