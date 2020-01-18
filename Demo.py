import numpy as np
import scipy.io as sio
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# 被试数
Nsubs = 10
# 设置时间滑窗
# 即单个时间点的计算选取前后共5个时间点的数据进行解码
timewindow = 5
# 重复计算次数
N = 10

for sub in range(Nsubs):

    # 读取EEG数据
    data = sio.loadmat("data.mat")["data"]
    # 读取标签信息
    label = np.loadtxt("label.txt")

    # 此时data的shape为[Nchannels, Ntimes, Ntrials]
    # Nchannels为导联数， Ntimes为采样的时间点数， Ntrials为试次数
    # label的shape为[Ntrials]

    Nchannels, Ntimes, Ntrials = data.shape

    # 实际计算的时间点（由于有时间滑窗的存在）
    Nts = Ntimes - timewindow + 1

    data = np.transpose(data, (2, 0, 1))

    # 初始化正确率（对应Nts个时间点）
    acc = np.zeros([Nts], dtype=np.float32)

    # 逐时间点解码
    for t in range(Nts):

        # 获取单个时间点的数据
        datat = data[:, t:t+timewindow, :]
        # 更改数据shape将试次信息提前
        datat = np.transpose(datat, (2, 0, 1))
        # 将每个试次的数据铺平
        datat = np.reshape(datat, [Ntrials, Nchannels*timewindow])

        # 初始化单个时间点的正确率（每个时间点经过N次重复计算）
        acct = np.zeros([N], dtype=np.float32)

        for i in range(N):
            # 将试次按4：1的比例随机分为训练集与测试集
            x_train, y_train, x_test, y_test = \
                train_test_split(datat, label, test_size=0.2)
            # 初始化线性SVM分类器
            svm = LinearSVC()
            # 分类器进行训练
            svm.fit(x_train, x_test)
            # 进行测试，得到改时间点单次分类的正确率
            acct[i] = svm.score(x_test, y_test)

        # 获取该时间点的解码正确率
        acc[t] = np.average(acct)

    # 将单个被试的解码正确率结果存为.txt文件
    np.savetxt("decodingACC"+str(sub)+".txt", acc)





