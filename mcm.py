"""mcm"""
import pandas as pd
def mcm(tn, fp, fn, tp):
    """Let be a confusion matrix like this:


      N    P
    +----+----+
    |    |    |
    | TN | FP |
    |    |    |
    +----+----+
    |    |    |
    | FN | TP |
    |    |    |
    +----+----+

    The observed values by columns and the expected values
    by rows and the positive class in right column.
    With these definitions, the TN, FP, FN and TP values are that order.


    Parameters
    ----------
    TN : integer
         True Negatives (TN) is the total number of outcomes where the model correctly predicts the negative class.
    FP : integer
         False Positives (FP) is the total number of outcomes where the model incorrectly predicts the positive class.
    FN : integer
         False Negatives (FN) is the total number of outcomes where the model incorrectly predicts the negative class.
    TP : integer
         True Positives (TP) is the total number of outcomes where the model correctly predicts the positive class.

    Returns
    -------
    sum : DataFrame
          DataFrame with several metrics

    Notes
    -----
    https://en.wikipedia.org/wiki/Confusion_matrix
    https://developer.lsst.io/python/numpydoc.html
    https://www.mathworks.com/help/risk/explore-fairness-metrics-for-credit-scoring-model.html

    Examples
    --------
    data = pd.DataFrame({
    'y_true': ['Positive']*47 + ['Negative']*18,
    'y_pred': ['Positive']*37 + ['Negative']*10 + ['Positive']*5 + ['Negative']*13})

    tn, fp, fn, tp = confusion_matrix(y_true = data.y_true,
                                  y_pred = data.y_pred,
                                  labels = ['Negative',
                                  'Positive']).ravel()

    """
    df_mcm = []

    df_mcm.append(['Sensitivity', tp / (tp + fn)])
    df_mcm.append(['Recall', tp / (tp + fn)])
    df_mcm.append(['True Positive rate (TPR)', tp / (tp + fn)])
    df_mcm.append(['Specificity', tn / (tn + fp)])
    df_mcm.append(['True Negative Rate (TNR)', tn / (tn + fp)])

    df_mcm.append(['Precision', tp / (tp + fp)])
    df_mcm.append(['Positive Predictive Value (PPV)', tp / (tp + fp)])
    df_mcm.append(['Negative Predictive Value (NPV)', tn / (tn + fn)])

    df_mcm.append(['False Negative Rate (FNR)', fn / (fn + tp)])
    df_mcm.append(['False Positive Rate (FPR)', fp / (fp + tn)])
    df_mcm.append(['False Discovery Rate (FDR)', fp / (fp + tp)])

    df_mcm.append(['Rate of Positive Predictions (PRR)'], (fp + tp) / (tn + tp + fn + fp))
    df_mcm.append(['Rate of Negative Predictions (RNP)'], (fn + tn) / (tn + tp + fn + fp))

    df_mcm.append(['Accuracy', (tp + tn) / (tp + tn + fp + fn)])
    df_mcm.append(['F1 Score', 2*tp / (2*tp + fp + fn)])

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)

    df_mcm.append(['Positive Likelihood Ratio (LR+)', tpr / fpr])
    df_mcm.append(['Negative Likelihood Ratio (LR-)', fnr / tnr])

    return pd.DataFrame(df_mcm, columns = ['Metric', 'Value'])
