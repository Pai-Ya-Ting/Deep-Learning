#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchaudio


# In[3]:


import os
import pandas as pd
import librosa
import numpy as np

df = pd.read_csv('train.csv',  dtype = {'file': str, 'scenario': str, 'sentence':str})

label = {}

for i, category in enumerate(np.unique(df['scenario'])):
    label[category] = i
    
df['label'] = df['scenario'].apply(lambda x: label[x])


# In[4]:


def read_audio(file_id, mode):
    if mode == 'train':
        filename = os.path.join(os.path.abspath('train/')+str('/' + file_id.file)+'.wav')
    elif mode == 'test':
        filename = os.path.join(os.path.abspath('test/')+'/'+str(file_id).zfill(6)+'.wav')
    print(filename)
    y, sr = librosa.load(filename, sr = 8000)
    if 0 < len(y): 
        y, _ = librosa.effects.trim(y)
#         y = librosa.util.fix_length(y, int(5*sr))
#         mfcc = librosa.feature.mfcc(y=y, n_mfcc = 40)
    return y

X = []
y = []
for i in df.itertuples():
    print(i.file)
    x = read_audio(i, 'train')
    x = librosa.util.fix_length(x, int(5*8000))
#     print(x.shape)
    X.append(x)
    y.append(i.label)
# choice = np.random.random_sample()
# X.append(x)
# y.append(i.label)
#     if choice < 0.3:
#         noise = np.random.randn(len(x))
#         x = x + 0.01 * noise
# #     X.append(augmented_data)
# #     y.append(i.label)
    
#     elif choice < 0.6:
#         x = librosa.effects.pitch_shift(x, 8000, n_steps=4)
#     X.append(x_pitch)
#     y.append(i.label)
    
#     elif choice < 0.6:
#         x = librosa.effects.time_stretch(x, 2.0)
# #         X.append(x_fast)
# #         y.append(i.label)
#     elif choice < 0.8:
#         x = librosa.effects.time_stretch(x, 0.5)


# X = np.array(X, dtype = object)
# y = np.array(y, dtype = object)

# np.save('y_m5',y)
# np.save('X_train_m5_v1',X)


# # print(np.array(X).shape)

X_test = []
for i in range(4721):
    print(i)
    x = read_audio(i, 'test')
    x = librosa.util.fix_length(x, int(5*8000))
    X_test.append(x)
    
# X_test = np.array(X_test, dtype = object)
# np.save('X_test_m5',X_test)
# # np.save('MFCC_40_x_test', X)
# print(np.array(X_test).shape)
X = np.array(X)#, dtype = object)
X_test = np.array(X_test)#, dtype = object)
y = np.array(y)
print(X.shape, X_test.shape)


# In[5]:


# X_test = np.load('X_test_raw.npy', allow_pickle=True).astype(float)
# X = np.load('X_train_raw.npy', allow_pickle=True).astype(float)
# y = np.load('y_raw.npy', allow_pickle=True).astype(int)


# In[6]:


# np.save('y_raw',y)
# np.save('X_train_raw',X)
# np.save('X_test_raw',X_test)


# In[7]:


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify = y)


# In[5]:


# X_train = np.load('X_train_0627_v1.npy', allow_pickle=True)
# y_train = np.load('y_train_0627_v1.npy', allow_pickle=True)
# X_val = np.load('X_val_0627_v1.npy', allow_pickle=True)
# y_val = np.load('y_val_0627_v1.npy', allow_pickle=True)


# In[6]:


import copy
X_tmp = copy.deepcopy(X_train)
x_tmp = []

for i in range(X_tmp.shape[0]):
    print(i)
    noise = np.random.randn(len(X_tmp[i]))
    x = X_tmp[i] + 0.001 * noise
    x = librosa.util.fix_length(x, int(5*8000))
    x_tmp.append(x)
    
    x = librosa.effects.pitch_shift(X_tmp[i], 8000, n_steps=4)
    x = librosa.util.fix_length(x, int(5*8000))
    x_tmp.append(x)
    
    x = librosa.effects.time_stretch(X_tmp[i], 1.5)
    x = librosa.util.fix_length(x, int(5*8000))
    x_tmp.append(x)
    
    x = librosa.effects.time_stretch(X_tmp[i], 0.5)
    x = librosa.util.fix_length(x, int(5*8000))
    x_tmp.append(x)

    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
    start = int(X_tmp[i].shape[0] * timeshift_fac)

    if (start > 0):
        x = np.pad(X_tmp[i],(start,0),mode='constant')[0:X_tmp[i].shape[0]]
    else:
        x = np.pad(X_tmp[i],(0,-start),mode='constant')[0:X_tmp[i].shape[0]]
    x = librosa.util.fix_length(x, int(5*8000))
    x_tmp.append(x)

    dyn_change = np.random.uniform(low=1.5,high=3)
    x = X_tmp[i] * dyn_change
    x = librosa.util.fix_length(x, int(5*8000))
    x_tmp.append(x)

X_train = np.concatenate([X_train, x_tmp])#.shape
y_train = np.concatenate([y_train, np.repeat(y_train, 6)])
print(X_train.shape, y_train.shape)


# In[12]:


# np.save('X_train_0627_v2_feature', X_train)
# np.save('y_train_0627_v2', y_train)


# In[13]:


# np.save('X_val_0627_v2_feature', X_val)
# np.save('y_val_0627_v2', y_val)


# In[10]:


# np.save('X_test_0627_v1_feature', X_test)


# In[8]:


# X_train = np.load('X_train_0627_v2_feature.npy', allow_pickle=True)
# y_train = np.load('y_train_0627_v2.npy', allow_pickle=True)
# X_val = np.load('X_val_0627_v2_feature.npy', allow_pickle=True)
# y_val = np.load('y_val_0627_v2.npy', allow_pickle=True)


# In[121]:


X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
train_dataset = TensorDataset(X_train, y_train)
X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 128, shuffle=True)
test_loader = DataLoader(X_test, batch_size = len(X_test), shuffle = False)


# In[162]:


class M5(nn.Module):
#     def __init__(self, n_input=1, n_output=10, stride=64, n_channel=128):
    def __init__(self, n_input=1, n_output=10, stride=32, n_channel=128):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=40, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel) 
        self.pool1 = nn.AvgPool1d(4) #AvgPool
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.AvgPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.AvgPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.AvgPool1d(4)
        
        self.conv5 = nn.Conv1d(2 * n_channel, n_channel, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.pool5 = nn.AvgPool1d(4)
        
#         n = int(n_channel/2)
#         self.conv6 = nn.Conv1d(n_channel, n, kernel_size=1)
#         self.bn6 = nn.BatchNorm1d(n)
#         self.pool6 = nn.AvgPool1d(4)
        
#         self.conv7 = nn.Conv1d(n, int(n/2), kernel_size=1)
#         self.bn7 = nn.BatchNorm1d(int(n/2))
#         self.pool7 = nn.AvgPool1d(4)
#         self.fc1 = nn.Linear(2 * n_channel, n_output)
        
        self.fc = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
#         x = F.dropout(x, p=0.05)
#         x = self.bn1(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.dropout(x, p=0.1)
#         x = self.bn2(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        x = F.relu(x)
        x = self.pool2(x)
#         x = F.dropout(x, p=0.05)
        x = self.conv3(x)
#         x = self.bn3(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        
        x = F.relu(x)
        x = self.pool3(x)
        x = F.dropout(x, p=0.1)
        
        x = self.conv4(x)
#         x = self.bn4(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        x = F.relu(x)
    
        x = self.pool4(x)
        x = F.dropout(x, p=0.05)
        
        x = self.conv5(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        x = F.relu(x)
    
        x = self.pool5(x)
        x = F.dropout(x, p=0.05)
        
#         x = self.conv6(x)
#         m = nn.LayerNorm(x.size()[1:])
#         x = m(x)
#         x = F.relu(x)
#         x = self.pool6(x)
#         x = F.dropout(x, p=0.05)
        
#         x = self.conv7(x)
#         m = nn.LayerNorm(x.size()[1:])
#         x = m(x)
#         x = F.relu(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = F.dropout(x, p=0.05)

#         x = self.fc1(x)
        return F.log_softmax(x, dim=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = M5(n_input=1, n_output=len(np.unique(y_train)))
model.to(device)
print(model)


# In[163]:


optimizer = optim.AdamW(model.parameters(), lr=1e-3)#, weight_decay=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)# weight_decay=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  


# In[164]:


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device).unsqueeze(1)
        target = target.to(device)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.cross_entropy(output.squeeze(), target)
#  cross_entropy nll_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} {np.mean(losses):.6f}")
            
        losses.append(loss.item())

def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in val_loader:

        data = data.to(device).unsqueeze(1)
        target = target.to(device)
        output = model(data)

        pred = output.argmax(dim = 2)
        correct += (pred.squeeze().eq(target).sum().item())

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n")


# In[165]:


log_interval = 20
n_epoch = 25
losses = []
# 25 + 5
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)
    test(model, epoch)
    print("="*60)
#     scheduler.step()


# In[166]:


with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device).unsqueeze(1)
        output = model(data)
    pred = output.argmax(dim=2)


# In[167]:


inv_label = {}
for i, j in label.items():
    inv_label.update({j:i})


# In[168]:


ans = pd.DataFrame(columns=['Category'])
ans['Category'] = list(map(lambda x: inv_label[x.item()], pred))
ans.insert(0, column="File", value = ans.index.values)
ans['Category'].value_counts(normalize=True)


# In[169]:


ans


# In[170]:


ans.iloc[9]


# In[171]:


ans.to_csv('ans_m5_3.csv', index=False)


# In[ ]:




