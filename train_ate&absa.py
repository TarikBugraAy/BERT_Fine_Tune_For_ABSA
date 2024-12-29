# %% [markdown]
# # import

# %%
from model.bert import bert_ATE, bert_ABSA
from data.dataset import dataset_ATM, dataset_ABSA

# %%
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrain_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
lr = 2e-5
model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)

# %%
def evl_time(t):
    min, sec= divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)

def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model
    
def save_model(model, name):
    torch.save(model.state_dict(), name)

# %% [markdown]
# # Acpect Term Extraction

# %%
laptops_train_ds = dataset_ATM(pd.read_csv("data/laptops_train.csv"), tokenizer)
laptops_test_ds = dataset_ATM(pd.read_csv("data/laptops_test.csv"), tokenizer)
restaurants_train_ds = dataset_ATM(pd.read_csv("data/restaurants_train.csv"), tokenizer)
restaurants_test_ds = dataset_ATM(pd.read_csv("data/restaurants_test.csv"), tokenizer)
twitter_train_ds = dataset_ATM(pd.read_csv("data/twitter_train.csv"), tokenizer)
twitter_test_ds = dataset_ATM(pd.read_csv("data/twitter_test.csv"), tokenizer)

# %%
# w,x,y,z = laptops_train_ds.__getitem__(121)
# print(w)
# print(x)
# print(x.size())
# print(y)
# print(y.size())
# print(z)
# print(z.size())

# %%
train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])
test_ds = ConcatDataset([laptops_test_ds, restaurants_test_ds, twitter_test_ds])

# %%
def create_mini_batch(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    tags_tensors = [s[2] for s in samples]
    tags_tensors = pad_sequence(tags_tensors, batch_first=True)

    pols_tensors = [s[3] for s in samples]
    pols_tensors = pad_sequence(pols_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
    
    return ids_tensors, tags_tensors, pols_tensors, masks_tensors

# %%
train_loader = DataLoader(train_ds, batch_size=5, collate_fn=create_mini_batch, shuffle = True)
test_loader = DataLoader(test_ds, batch_size=50, collate_fn=create_mini_batch, shuffle = True)

# %%
# for batch in train_loader:
#     w,x,y,z = batch
#     print(w)
#     print(w.size())
#     print(x)
#     print(x.size())
#     print(y)
#     print(y.size())
#     print(z)
#     print(z.size())
#     break

# %%
def train_model_ATE(loader, epochs):
    all_data = len(loader)
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []
        correct_predictions = 0
        
        for data in loader:
            t0 = time.time()
            ids_tensors, tags_tensors, _, masks_tensors = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            loss = model_ATE(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors)
            losses.append(loss.item())
            loss.backward()
            optimizer_ATE.step()
            optimizer_ATE.zero_grad()

            finish_data += 1
            current_times.append(round(time.time()-t0,3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
            print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         

        save_model(model_ATE, 'bert_ATE.pkl')
        
def test_model_ATE(loader):
    pred = []
    trueth = []
    with torch.no_grad():
        for data in loader:

            ids_tensors, tags_tensors, _, masks_tensors = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            outputs = model_ATE(ids_tensors=ids_tensors, tags_tensors=None, masks_tensors=masks_tensors)

            _, predictions = torch.max(outputs, dim=2)

            pred += list([int(j) for i in predictions for j in i ])
            trueth += list([int(j) for i in tags_tensors for j in i ])

    return trueth, pred



# %%
%time train_model_ATE(train_loader, 3)

# %%
model_ATE = load_model(model_ATE, 'bert_ATE.pkl')

# %%
%time x, y = test_model_ATE(test_loader)
print(classification_report(x, y, target_names=[str(i) for i in range(3)]))

# %% [markdown]
# # Aspect Based Sentiment Analysis

# %%
laptops_train_ds = dataset_ABSA(pd.read_csv("data/laptops_train.csv"), tokenizer)
laptops_test_ds = dataset_ABSA(pd.read_csv("data/laptops_test.csv"), tokenizer)
restaurants_train_ds = dataset_ABSA(pd.read_csv("data/restaurants_train.csv"), tokenizer)
restaurants_test_ds = dataset_ABSA(pd.read_csv("data/restaurants_test.csv"), tokenizer)
twitter_train_ds = dataset_ABSA(pd.read_csv("data/twitter_train.csv"), tokenizer)
twitter_test_ds = dataset_ABSA(pd.read_csv("data/twitter_test.csv"), tokenizer)

# %%
w,x,y,z = laptops_train_ds.__getitem__(121)
print(w)
print(len(w))
print(x)
print(len(x))
print(y)
print(len(y))
print(z)

# %%
def create_mini_batch2(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    segments_tensors = [s[2] for s in samples]
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    label_ids = torch.stack([s[3] for s in samples])
    
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

    return ids_tensors, segments_tensors, masks_tensors, label_ids

# %%
train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])
test_ds = ConcatDataset([laptops_test_ds, restaurants_test_ds, twitter_test_ds])

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=create_mini_batch2, shuffle = True)
test_loader = DataLoader(test_ds, batch_size=50, collate_fn=create_mini_batch2, shuffle = True)

# %%
# for batch in train_loader:
#     w,x,y,z = batch
#     print(w)
#     print(w.size())
#     print(x)
#     print(x.size())
#     print(y)
#     print(y.size())
#     print(z)
#     print(z.size())
#     break

# %%
def train_model_ABSA(loader, epochs):
    all_data = len(loader)
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []
        correct_predictions = 0
        
        for data in loader:
            t0 = time.time()
            ids_tensors, segments_tensors, masks_tensors, label_ids = data
            ids_tensors = ids_tensors.to(DEVICE)
            segments_tensors = segments_tensors.to(DEVICE)
            label_ids = label_ids.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            loss = model_ABSA(ids_tensors=ids_tensors, lable_tensors=label_ids, masks_tensors=masks_tensors, segments_tensors=segments_tensors)
            losses.append(loss.item())
            loss.backward()
            optimizer_ABSA.step()
            optimizer_ABSA.zero_grad()

            finish_data += 1
            current_times.append(round(time.time()-t0,3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
            print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         

        save_model(model_ABSA, 'bert_ABSA2.pkl')
        
def test_model_ABSA(loader):
    pred = []
    trueth = []
    with torch.no_grad():
        for data in loader:

            ids_tensors, segments_tensors, masks_tensors, label_ids = data
            ids_tensors = ids_tensors.to(DEVICE)
            segments_tensors = segments_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            outputs = model_ABSA(ids_tensors, None, masks_tensors=masks_tensors, segments_tensors=segments_tensors)
            
            _, predictions = torch.max(outputs, dim=1)

            pred += list([int(i) for i in predictions])
            trueth += list([int(i) for i in label_ids])

    return trueth, pred



# %%
%time train_model_ABSA(train_loader, 6)

# %%
model_ABSA = load_model(model_ABSA, 'bert_ABSA.pkl')

# %%
%time x, y = test_model_ABSA(test_loader)
print(classification_report(x, y, target_names=[str(i) for i in range(3)]))

# %% [markdown]
# # ATE + ABSA

# %%
def predict_model_ABSA(sentence, aspect, tokenizer):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)

    word_pieces = ['[cls]']
    word_pieces += t1
    word_pieces += ['[sep]']
    word_pieces += t2

    segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

    with torch.no_grad():
        outputs = model_ABSA(input_tensor, None, None, segments_tensors=segment_tensor)
        _, predictions = torch.max(outputs, dim=1)
    
    return word_pieces, predictions, outputs

def predict_model_ATE(sentence, tokenizer):
    word_pieces = []
    tokens = tokenizer.tokenize(sentence)
    word_pieces += tokens

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model_ATE(input_tensor, None, None)
        _, predictions = torch.max(outputs, dim=2)
    predictions = predictions[0].tolist()

    return word_pieces, predictions, outputs

def ATE_ABSA(text):
    terms = []
    word = ""
    x, y, z = predict_model_ATE(text, tokenizer)
    for i in range(len(y)):
        if y[i] == 1:
            if len(word) != 0:
                terms.append(word.replace(" ##",""))
            word = x[i]
        if y[i] == 2:
            word += (" " + x[i])
            
    
    if len(word) != 0:
            terms.append(word.replace(" ##",""))
            
    print("tokens:", x)
    print("ATE:", terms)
    
    if len(terms) != 0:
        for i in terms:
            _, c, p = predict_model_ABSA(text, i, tokenizer)
            print("term:", [i], "class:", [int(c)], "ABSA:", [float(p[0][0]), float(p[0][1]), float(p[0][2])])


# %%
model_ABSA = load_model(model_ABSA, 'bert_ABSA.pkl')
model_ATE = load_model(model_ATE, 'bert_ATE.pkl')

# %%
text = "For the price you pay this product is very good. However, battery life is a little lack-luster coming from a MacBook Pro."
ATE_ABSA(text)

# %%
text = "I think Apple is better than Microsoft."
ATE_ABSA(text)

# %% [markdown]
# # Cyberpunk 2077 - Xbox One
# 
# https://www.amazon.com/-/zh_TW/Cyberpunk-2077-Xbox-One/product-reviews/B07DJW4WZC/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2

# %%
text = "Spent 5 hours downloading updates."
ATE_ABSA(text)

# %%
text = "Install is buggy, so after downloading a day one patch that's nearly 3 times the size of the game, it glitched on the CDs and had to reinstall the game from scratch."
ATE_ABSA(text)

# %%
text = "Cyberpunk 2077 freezes constantly, frame rates are terrible, and it's extremely frustrating to try to play."
ATE_ABSA(text)

# %%
text = "Cyberpunk 2077 is completely unplayable on xbox one. They should have never released this for current gen."
ATE_ABSA(text)

# %%
text = "It’s just a cash grab, the game crashes constantly, runs at like 20 fps, half the environment and characters only load when you’re three feet away from them. Unless you’re in a small space the game looks awful. The worst game i’ve ever played in years visually. It looks worse than later xbox 360 games."
ATE_ABSA(text)

# %%
text = "CD Projekt Red should have just abandoned the current gen consoles instead of cheating people out of their money."
ATE_ABSA(text)

# %%



