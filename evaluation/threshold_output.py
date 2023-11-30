import pandas as pd
from torch import tensor
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryFBetaScore, BinaryAUROC

threshold = 0.4

acc = BinaryAccuracy(threshold=threshold)
precision = BinaryPrecision(threshold=threshold)
recall = BinaryRecall(threshold=threshold)
fB = BinaryFBetaScore(beta=2., threshold=threshold)
auroc = BinaryAUROC()

df_output = pd.read_csv('../Kfolds/output/SD/08.11_10.16/fold_1/pubmed_abstract_08.11_10.32_epoch6.csv')

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
auroc_value = auroc(output, test_label)

print(
    f"recall:{recall5:.4f} precision:{precision5:.4f} fBeta:{fB5:.4f} acc:{acc5:.4f} auroc:{auroc_value:.4f}")
