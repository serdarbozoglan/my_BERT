import torch.nn as nn
import torch
import transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from scipy import stats
import torch_xla.core.xla_model as xm # for gpu or tpu?

# https://www.kaggle.com/c/google-quest-challenge/data -->dataset
class BERTBaseUncased(nn.Module):

    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30) # There are 30 class in the dataset


    def forward(self, ids, mask, segment_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=segment_ids) # first one is sequential output and second (o2) is pooled output
        bo = self.bert_drop(o2) # bert output
        return self.out(bo)

    class BERTDatasetTraining:
        def __init__(self,qtitle, qbody, answer, targets, tokenizer, max_len): #qtitle --> question title, qbody --> question body from dataset
            self.qtitle = qtitle
            self.qbody = qbody
            self.answer = answer
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.targets = targets # targets is a numpy array

        def __len__(self):
            return len(self.answer)

        def __getitem__(self, item):
            question_title = str(self.qtitle[item])
            question_body = str(self.qbody[item])
            answer = str(self.answer[item])

            #[CLS] [Q-TITLE] [Q-BODY] [SEP] [ANSWER] [SEP]

            encoded_dict = self.tokenizer.encode_plus(
                        question_title + " " + question_body,
                        answer,
                        add_special_tokens = True,
                        max_len = self.max_len
                        )
            
            ids = encoded_dict['input_ids']
            segment_ids = encoded_dict['token_type_ids']
            mask = encoded_dict['attention_mask']

            padding_len = self.max_len - len(ids)
            ids = ids + ([0] * padding_len)
            segment_ids = segment_ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)

            return {
                "ids" : torch.tensor(ids, dtype=torch.long),
                "mask" : torch.tensor(mask, dtyep=torch.long),
                "segment_ids" : torch.tensor(segment_ids, dtype=torch.long),
                "targets" : torch.tensor(self.targets[item, :], dtype=torch.float)
            }


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)
        
def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader): # bi --> batch index
        ids = d['ids']
        maks = d['mask']
        segment_ids = d['segment_ids']
        targets  = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        segment_ids = segment_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=segment_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True) # optimizer.step()'in yerine kullaniyoruz
        if scheduler is not None:
            scheduler.step()
        if bi % 10 == 0:
            print(f"bi={bi}, loss={loss}")

def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader): # bi --> batch index
        ids = d['ids']
        maks = d['mask']
        segment_ids = d['segment_ids']
        targets  = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        segment_ids = segment_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, mask=mask, token_type_ids=segment_ids)
        loss = loss_fn(outputs, targets)
        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())

    return np.vstack(fin_outputs), np.vstack(fin_targets)


def run():
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    EPOCHS = 20

    df = pd.read_csv("../inputs/train.csv").fillna("none")
    df_train, df_valid = train_test_split(df, random_state=4299, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    sample = pd.read_csv("../input/sample_submission.csv")
    target_cols = list(sample.drop("qa_id", axis-1).columns)
    train_targets = df_train[target_cols].to_numpy() # alternative is .values
    valid_targets = df_valid[target_cols].to_numpy()

    tokenizer = transformers.BertTokenizer.from_pretrained("/Users/serdar/DATASETS/bert-base-uncased/")

    train_dataset = BERTDatasetTraining(
        qtitle = df_train.question_title.to_numpy(),
        qbody = df_train.question_body.to_numpy(),
        asnwer = df_train.answer.to_numpy(),
        targets = train_targets,
        tokenizer = tokenizer,
        max_len = MAX_LEN        
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True
    )

    valid_dataset = BERTDatasetTraining(
        qtitle = df_valid.question_title.to_numpy(),
        qbody = df_valid.question_body.to_numpy(),
        asnwer = df_valid.answer.to_numpy(),
        targets = valid_targets,
        tokenizer = tokenizer,
        max_len = MAX_LEN        
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = 4,
        shuffle = False # does not matter shuffling for validation but training
    )

    device = xm.xla_device() # makes cpu/gpu compatible
    lr = 3e-5 # learning rate
    num_train_steps = int(len(train_dataset)/ TRAIN_BATCH_SIZE * EPOCHS)
    model = BERTBaseUncased("/Users/serdar/DATASETS/bert_based_uncased/").to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=num_train_steps,

    )

    for epoch in range(EPOCHS):
        train_loop_fn(train_data_loader, model, optimizer, device, scheduler)
        t, o = eval_loop_fn(valid_data_loader, model, device) # targets, outputs

        spear = [] # this is for kaggle competition
        for jj in range(t.shape[1]):
            p1 = list(t[:, jj])
            p2 = list(o[:, jj])
            coef, _ = np.nan_to_num(stats.spearmanr(r1, r2))
            spear.append(coef)
        spear = np.mean(spear)
        print(f"epoch ={epoch}, spearman={spear}")
        xm.save(model.state_dict(), "model.bin") # it will be cpu/gpu compatible

if __name__=="__main__":
    run()







