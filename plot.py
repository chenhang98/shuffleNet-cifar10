from matplotlib import pyplot as plt
import numpy as np  
import csv

def load(fn):
    reader = csv.reader(open(fn))
    iters, acc, loss = [], [], []
    val_acc, val_loss = [], []

    names = next(reader)
    for row in reader:        
        loss.append(float(row[1]))
        val_loss.append(float(row[2]))
        
        acc.append(float(row[3]))
        val_acc.append(float(row[4]))

    return np.array(loss), np.array(val_loss), np.array(acc), np.array(val_acc)


def watch(fn, acfunc = lambda x: x, lsfunc = lambda x: x):
    loss, val_loss, acc, val_acc = load(fn)
    
    print("last acc/val_acc: ", acc[-1], val_acc[-1])  
    print("max acc/val_acc: ", max(acc), max(val_acc))

    plt.subplot(211)
    plt.plot(acfunc(acc), 'b', label = "acc")
    plt.plot(acfunc(val_acc), 'r', label = "val_acc")
    plt.legend()

    plt.subplot(212)
    plt.plot(lsfunc(loss[2:]), 'b', label = "loss")
    plt.plot(lsfunc(val_loss[2:]), 'r', label = "val_loss")
    plt.legend()
    plt.show()


def compare(fp1, fp2, acfunc = lambda x: x, lsfunc = lambda x: x, 
    labels1 = ["acc1", "val_acc1", "loss1", "val_loss1"], labels2 = ["acc2", "val_acc2", "loss2", "val_loss2"]):
    loss1, val_loss1, acc1, val_acc1 = load(fp1)
    loss2, val_loss2, acc2, val_acc2 = load(fp2)
    
    plt.subplot(211)
    plt.plot(acfunc(acc1), 'b', label = labels1[0]) 
    plt.plot(acfunc(val_acc1), 'r', label = labels1[1])
    plt.plot(acfunc(acc2), 'g', label = labels2[0])
    plt.plot(acfunc(val_acc2), 'y', label = labels2[1])
    plt.ylabel("")
    plt.xlabel("iteration")
    plt.ylabel("log")
    # plt.ylim(-3.5, -1)

    plt.legend()
    
    plt.subplot(212)
    plt.plot(lsfunc(loss1[1:]), 'b', label = labels1[2])
    plt.plot(lsfunc(val_loss1[1:]), 'r', label = labels1[3])
    plt.plot(lsfunc(loss2[1:]), 'g', label = labels2[2])
    plt.plot(lsfunc(val_loss2[1:]), 'y', label = labels2[3])
    plt.xlabel("iteration")
    plt.ylabel("log")
    # plt.ylim(-2, 0)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # compare("log_05.csv", "log.csv", acfunc = lambda x: np.log(1 - x), lsfunc = np.log,
        # labels1 = ["err", "val_err", "loss", "val_loss"], labels2 = ["err SE", "val_err SE", "loss SE", "val_loss SE"])
    # compare("log_05.csv", "log.csv")
    watch("log.csv")

# scale factor      accuracy    time        g       size/M      flops/M
#   1.0             92.29       11.4h       8       0.9131      161.70
#   0.5             91.48       6.5h        8       0.2507      43.43
#   0.5             92.60       4.0h        3       0.2427      42.97
#   0.5             91.44       3.6h        1       0.2487      44.63

# resnet 20         91.25                           0.27
# Deep Residual Learning for Image Recognition