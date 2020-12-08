import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from efficientnet_pytorch import EfficientNet
from math import ceil



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb
DATA_DIR = os.getcwd()+"/npy_100/Train200"
DATA_DIR_Test = os.getcwd()+"/npy_100/Test60"
def stack(path,label_names, frame=20):
    x,y=[],[]
    for f1 in label_names:
        path_loop = os.path.join(path, f1)
        for f2 in [f2 for f2 in os.listdir(path_loop) if f2[-4:] == ".npy"]:
            a = np.load(path_loop + "/" + f2)
            # a = get_rgb(a)
            num1 = len(a)//frame
            num2 = ceil(len(a)/frame)
            a_recrop=[]
            if (num1==num2):
                for i in range(num1):
                    a_clip = a[i*frame:(i+1)*frame]
                    a_recrop.append(a_clip)
            if (num1 !=num2):
                for i in range(num1):
                    a_clip = a[i * frame:(i + 1) * frame]
                    a_recrop.append(a_clip)
                a_recrop.append(a[len(a)-frame:])
            a = np.asarray(a_recrop)
            a = a.transpose(0, 4, 1, 2, 3)/255
            a = a.reshape(len(a),5,20,-1)
            x.append(a)
            temp1 = np.ones(len(a))
            temp2 = np.zeros(len(a))
            if f1 == "Violence":
                y_temp = np.dstack((temp1, temp2)).reshape(-1, 2)
            elif f1 == "NoViolence":
                y_temp = np.dstack((temp2, temp1)).reshape(-1, 2)
            y.append(y_temp)
    y = np.vstack(y)
    x = np.vstack(x)
    return x, y



label_names = ["Violence", "NoViolence"]
x_train, x_test, y_train, y_test=[],[],[],[]
x_train, y_train = stack(DATA_DIR, label_names)
x_test, y_test = stack(DATA_DIR_Test, label_names)
print(x_train.shape,y_train.shape)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train,y_train = torch.tensor(x_train).view(len(x_train),5,20,10000).float().to(device),torch.tensor(y_train).to(device)
x_test,y_test = torch.tensor(x_test).view(len(x_train),5,20,10000).float().to(device),torch.tensor(y_test).to(device)
print(x_train.shape,y_train.shape)

LR = 1e-3
N_EPOCHS = 500
BATCH_SIZE = 100
DROPOUT = 0.2


def calculate_acuracy_mode_one(model_pred, labels):
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0
    true_predict_num = torch.sum(pred_result * labels)
    precision = true_predict_num / pred_one_num
    return precision.item()


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5,
                               out_channels=3,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.efficientNet = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.efficientNet._fc.out_features
        self.linear0_bn = nn.BatchNorm1d(num_ftrs)
        self.linear1 = nn.Linear(num_ftrs, 512)
        self.linear1_bn = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(512, 256)
        self.linear2_bn = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 2)
        self.act = torch.relu


    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.linear0_bn(self.efficientNet(x)))
        x = self.act(self.linear1_bn(self.linear1(x)))
        x = self.drop(self.act(self.linear2_bn(self.linear2(x))))
        return self.linear3(x)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=LR, weight_decay=0.5*LR)
scheduler_adam_p1= torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma = 0.1)
scheduler_adam_p2= torch.optim.lr_scheduler.StepLR(optimizer,step_size=8, gamma = 0.5)
scheduler_adam_p3= torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma = 0.8)
criterion=nn.BCEWithLogitsLoss()
criterion_test=nn.BCELoss()


# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
min_loss = 1000
loss_train_plot = []
loss_test_plot_relu=[]
acc_test_plot=[]
for epoch in range(N_EPOCHS):
    model.train()
    loss_train=0
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds].float())
        loss.backward()
        optimizer.step()
        loss_train+=loss.item()
    if (epoch >= 0 and epoch < 6):
        scheduler_adam_p1.step()
    if (epoch >= 0 and epoch < 14):
        scheduler_adam_p2.step()
    if (epoch >= 20 and epoch < 50):
        scheduler_adam_p3.step()
    model.eval()
    with torch.no_grad():
        y_test_pred = torch.tensor([])
        for batch in range(len(x_test) // BATCH_SIZE + 1):
            slices = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            model.eval()
            y_test_pred1 = model(x_test[slices])
            y_test_pred1=y_test_pred1.cpu()
            y_test_pred = torch.cat([y_test_pred, y_test_pred1], dim=0)
        y_test_pred_sigmoid = torch.sigmoid(y_test_pred)
        predicted_relu = (y_test_pred.data.cuda() > 0).float()
        predicted_sigmoid = (y_test_pred_sigmoid.data.cuda() > 0.5).float()
        loss_relu = criterion_test(predicted_relu , y_test.float())
        loss_sigmoid = criterion_test(predicted_sigmoid, y_test.float())
        loss_test_relu = loss_relu.item()
        loss_test_sigmoid=loss_sigmoid.item()
        loss_test_plot_relu.append(loss_test_relu)
        loss_train_plot.append(loss_train)
        acc_test_plot.append(calculate_acuracy_mode_one(predicted_relu,y_test))
        if loss_test_sigmoid < min_loss:
            min_loss = loss_test_sigmoid
            print("save model")
            torch.save(model.state_dict(), 'model_hongfei_niu.pt')
    print('Epoch {}: Relu loss: {:.5f} - Sig loss: {:.5f} - Relu Accuracy: {:.5f} - {}'.format(epoch+1, loss_test_relu,loss_test_sigmoid,calculate_acuracy_mode_one(predicted_relu,y_test),optimizer.param_groups[0]['lr']))



plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(epoch)+1, np.asarray(loss_train_plot), label="train loss")
plt.plot(np.arange(epoch)+1, np.asarray(loss_test_plot_relu), label="val_loss")
plt.plot(np.arange(epoch)+1, acc_test_plot, label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.title("Training Loss and Accuracy on dataset")
plt.legend(loc = 3)
plt.show()


DROPOUT=0
def predict(x):
    # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=5,
                                   out_channels=3,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.efficientNet = EfficientNet.from_pretrained('efficientnet-b0')
            num_ftrs = self.efficientNet._fc.out_features
            self.linear0_bn = nn.BatchNorm1d(num_ftrs)
            self.linear1 = nn.Linear(num_ftrs, 512)
            self.linear1_bn = nn.BatchNorm1d(512)
            self.drop = nn.Dropout(DROPOUT)
            self.linear2 = nn.Linear(512, 256)
            self.linear2_bn = nn.BatchNorm1d(256)
            self.linear3 = nn.Linear(256, 2)
            self.act = torch.relu

        def forward(self, x):
            x = self.conv1(x)
            x = self.act(self.linear0_bn(self.efficientNet(x)))
            x = self.act(self.linear1_bn(self.linear1(x)))
            x = self.drop(self.act(self.linear2_bn(self.linear2(x))))
            return self.linear3(x)

    model = Net().cpu()
    model.load_state_dict(torch.load("model_hongfei_niu.pt"))
    y_pred = torch.tensor([])
    for batch in range(len(x) // BATCH_SIZE + 1):
        model.eval()
        slices = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        y_test_pred = model(x[slices].cpu()).detach()
        y_test_pred = torch.sigmoid(y_test_pred)
        y_test_pred = (y_test_pred.data.cpu() > 0.5).float()
        y_pred = torch.cat([y_pred, y_test_pred], dim=0)
    return y_pred


y_pred = predict(x_test)
loss_final = criterion_test(y_pred, y_test.float().cpu())
acc = calculate_acuracy_mode_one(y_pred,y_test.float().cpu())
predict=np.argmax(y_pred.cpu(),axis=1)
raw=np.argmax(y_test.cpu(),axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred = predict,y_true = raw)
print('confusion matrix')
print(cm)

