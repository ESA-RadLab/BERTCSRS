import pandas as pd
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score, \
    BinaryCohenKappa, BinaryFBetaScore, BinaryPrecisionRecallCurve
from torch import tensor


threshold = 0.122

acc = BinaryAccuracy(threshold=threshold)
precision = BinaryPrecision(threshold=threshold)
recall = BinaryRecall(threshold=threshold)
fB = BinaryFBetaScore(beta=2., threshold=threshold)

df_output = pd.read_csv('output/biobert_04.08_09.29_epoch7.csv')

output = tensor(df_output['prediction'])

labels_dict = {
    'Excluded': 0,
    'Included': 1,
}

labels = [labels_dict[label] for label in df_output['decision']]
test_label = tensor(labels)

acc5 = acc(output, test_label)
precision5 = precision(output, test_label)
recall5 = recall(output, test_label)
fB5 = fB(output, test_label)

print(
    f"recall:{recall5:.4f} precision:{precision5:.4f} fBeta:{fB5:.4f} acc:{acc5:.4f}")
