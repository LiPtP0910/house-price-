import pandas as pd
import torch
from torch import nn
import torch.utils
import torch.utils.data
from d2l import torch as d2l
from torch.nn import init
import math
#讀資料
train_data=pd.read_csv('C:/Users/walter/OneDrive/桌面/收集/大學nn專題/nn實作/房價預測/train.csv')
test_data=pd.read_csv('C:/Users/walter/OneDrive/桌面/收集/大學nn專題/nn實作/房價預測/test.csv')

#預處理

#print(train_data.shape)
#.iloc為切割矩陣
#print(train_data.iloc[0:4,[0,-1]])


#train_data.iloc[:,1:-1]去第一行和最後一行
#concat為合併
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

""""
print(train_data.dtypes)
Id                 int64
MSSubClass         int64
MSZoning          object
LotFrontage      float64
"""
numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
#print(numeric_features)
all_features[numeric_features]=all_features[numeric_features].apply(lambda x : (x-x.mean())/x.std())
all_features[numeric_features]=all_features[numeric_features].fillna(0)
all_features=pd.get_dummies(all_features,dummy_na=True)
#print(all_features.shape)

n_train=train_data.shape[0]#shape[0]表示列數
#all_features[:n_train]同all_features[:n_train,:]為常見的簡潔寫法
#.values()為將DataFrame轉成Numpy數列(array)
train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float)
test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float)
train_labels=torch.tensor(train_data.SalePrice.values,dtype=torch.float).view(-1,1)

#模型訓練
loss=torch.nn.MSELoss()#均方損失函數
'''
def get_net(feature_num):
    net=nn.Linea(feature_num,1)
    for parm in net.parameters():
        nn.init.normal_(parm,mean=0,std=0.01)
    return net
'''
num_input,num_outputs,num_hiddens=331,1,10
# 定义神经网络模型
get_net = nn.Sequential(
    nn.Linear(num_input, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

# 初始化网络参数
for params in get_net.parameters():
    init.normal_(params, mean=0, std=0.01)
#這是比賽用來評價模型好壞的比較函數，不參與模型修改
#rmse=Root Mean Squared Error均方根誤差
def log_rmse(net,feature,labels):
    with torch.no_grad():#為上下文管理器，告訴torch這不用算梯度
        #將小於1的值另為1，使取對數時更穩定
        clipped_preds=torch.max(net(feature),torch.tensor(1.0))
        rmse=torch.sqrt(loss(clipped_preds.log(),labels.log()))
    return rmse.item()#從Pytorch張量轉成數值

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]

    #https://zhuanlan.zhihu.com/p/371516520
    dataset=torch.utils.data.TensorDataset(train_features,train_labels)#將標籤和特徵打包
    #DataLoader本質是一迭代器，批次處理data
    train_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)#shuffle=True為打亂dataset

    optimizer=torch.optim.Adam(params=net.parameters(),lr=learning_rate,weight_decay=weight_decay)#weight_decay為權重衰減
    net=net.float()#參數轉成float
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l=loss(net(X.float()),y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

#K折(份)交叉驗證
#建訓練集和驗證集
def get_k_fold_data(k,i,x,y):
    assert k>1 #assert為斷言，會檢查條件，不滿足則跳例外不執行
    fold_size=x.shape[0]//k#總行數分K份
    x_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx]
        if j==i:
            x_vaild,y_vaild=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=torch.cat((x_train,x_part),dim=0)
            y_train=torch.cat((y_train,y_part),dim=0)
    return x_train,y_train,x_vaild,y_vaild

def k_fold(k,x_train,y_train,num_epochs,learning_rate,weight_decay,batch_size,net):
    train_l_sum,valid_l_sum=0,0
    for i in range(k):
        data=get_k_fold_data(k,i,x_train,y_train)
        flattened_x_train= torch.flatten(x_train, start_dim=1)
        #net=get_net(flattened_x_train)
        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum+=train_ls[-1]
        valid_l_sum+=valid_ls[-1]
        #畫隨批次增加，模型的學習狀態，僅在第一折時畫圖以節省資源
        #if i ==0:
         #   d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse',range(1,num_epochs+1),valid_ls,['train','valid'])
            
        print('fold ',i,'train rmse ',train_ls[-1],'valid rmse',valid_ls[-1])
    return train_l_sum/k,valid_l_sum/k

k,num_epochs,lr,weight_decay,batch_size=5,100,5,0.05,64
train_l,valid_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size,get_net)
print(k,'-fold validation: avg train rmse ',train_l,'avg valid rmse ',valid_l)

def train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size):
    train_ls, _ =train(get_net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    print('train rmse',train_ls[-1])
    preds=get_net(test_features).detach().numpy()#derach()為建立一個新Tensor與原來的無關，以避免反向傳遞
    test_data['SalePrice']=pd.Series(preds.reshape(1,-1)[0])
    submission=pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('submission.csv',index=False)

train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)
 
            

