from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

def pr(n_classes,y, y_prob):
    # 绘制多分类PR曲线
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_prob[:, i])

    plt.rcParams['font.family'] = ['Times New Roman']
    plt.figure()

    precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(), y_prob.ravel())
    average_precision["micro"] = average_precision_score(y, y_prob, average="micro")

    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label="micro_average P_R(area={0:0.2f})".format(average_precision["micro"]))
    # for i in range(n_classes):
    #     plt.plot(recall[i], precision[i], label="P_R curve of class{0}(area={1:0.2f})".format(i, average_precision[i]))
    idx7 = np.argmin(abs(0.7-recall["micro"]))
    idx8 = np.argmin(abs(0.8-recall["micro"]))
    idx9 = np.argmin(abs(0.9-recall["micro"]))
    score = 0.5*precision["micro"][idx7] + 0.3*precision["micro"][idx8] + 0.2*precision["micro"][idx9]
    print("score:", score)

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision recall curve with score={}'.format(score))
    plt.legend(loc="lower left")
    plt.show()

