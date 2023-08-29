import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score, \
    BinaryCohenKappa, BinaryFBetaScore, BinaryPrecisionRecallCurve
from torch import tensor


def evaluate(bert, version, epoch, data_path):
    acc = BinaryAccuracy(threshold=0.5)
    acc_3 = BinaryAccuracy(threshold=0.3)
    acc_1 = BinaryAccuracy(threshold=0.2)
    precision = BinaryPrecision(threshold=0.5)
    precision_3 = BinaryPrecision(threshold=0.3)
    precision_1 = BinaryPrecision(threshold=0.2)
    recall = BinaryRecall(threshold=0.5)
    recall_1 = BinaryRecall(threshold=0.2)
    recall_3 = BinaryRecall(threshold=0.3)
    auroc = BinaryAUROC()
    fB = BinaryFBetaScore(beta=2., threshold=0.5)
    fB_3 = BinaryFBetaScore(beta=2., threshold=0.3)
    fB_1 = BinaryFBetaScore(beta=2., threshold=0.2)
    PRcurve = BinaryPrecisionRecallCurve()

    output_data_path = os.path.join(data_path, f"{bert}_{version}_epoch{epoch}.csv")

    df_output = pd.read_csv(output_data_path)

    output = tensor(df_output['prediction'])

    labels_dict = {
        'Excluded': 0,
        'Included': 1,
    }

    labels = [labels_dict[label] for label in df_output['decision']]
    test_label = tensor(labels)

    acc5 = acc(output, test_label)
    acc3 = acc_3(output, test_label)
    acc1 = acc_1(output, test_label)
    precision5 = precision(output, test_label)
    precision3 = precision_3(output, test_label)
    precision1 = precision_1(output, test_label)
    recall5 = recall(output, test_label)
    recall3 = recall_3(output, test_label)
    recall1 = recall_1(output, test_label)
    auroc_value = auroc(output, test_label)
    fB5 = fB(output, test_label)
    fB3 = fB_3(output, test_label)
    fB1 = fB_1(output, test_label)
    precisions, recalls, thresholds = PRcurve(output, test_label)

    recall_list = []
    precision_list = []
    threshold_list = []
    for i, recall_value in enumerate(recalls):
        precision_value = precisions[i]
        if recall_value >= 0.95 and precision_value > 0.2:
            recall_list.append(f"{recall_value.item():.4f}")
            precision_list.append(f"{precisions[i].item():.4f}")
            threshold_list.append(f"{thresholds[i].item():.4f}")

    # fig_, ax_ = PRcurve.plot()

    # display = PrecisionRecallDisplay.from_predictions(
    #     test_label, output, name="LinearSVC", plot_chance_level=True
    # )

    print(
        f"recall:{recall5:.4f} precision:{precision5:.4f} fBeta:{fB5:.4f} acc:{acc5:.4f} recall3:{recall3:.4f} "
        f"precision3:{precision3:.4f} fBeta3:{fB3:.4f} acc3:{acc3:.4f} recall2:{recall1:.4f} precision2:{precision1:.4f} "
        f"fBeta2:{fB1:.4f} acc2:{acc1:.4f} auroc:{auroc_value:.4f}")

    # precision_list = [f"{p:.3f}" for p in precision_list]
    # recall_list = [f"{p:.3f}" for p in recall_list]
    # threshold_list = [f"{p:.3f}" for p in threshold_list]

    print(f"PRcurve:\nprecision : {precision_list} \nrecall    : {recall_list} \nthresholds: {threshold_list}")

    # plt.show()
    return precision_list, recall_list, threshold_list