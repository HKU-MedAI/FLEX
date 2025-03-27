import pandas as pd
import numpy as np
import cvxpy as cp

def generate(data, category, values, crossfolds = 5, target_column = 'CV5', patient_column = 'submitter_id', site_column = 'SITE', timelimit = 100, randomseed=0):
    ''' Generates 3 site preserved cross folds with optimal stratification of category
    Input:
        data: dataframe with slides that must be split into crossfolds.
        category: the column in data to stratify by
        values: a list of possible values within category to include for stratification
        crossfolds: number of crossfolds to split data into
        target_column: name for target column to contain the assigned crossfolds for each patient in the output dataframe
        patient_column: column within dataframe indicating unique identifier for patient
        site_column: column within dataframe indicating designated site for a patient
        timelimit: maximum time to spend solving
    Output:
        dataframe with a new column, 'CV3' that contains values 1 - 3, indicating the assigned crossfold
    '''

    submitters = data[patient_column].unique()
    newData = pd.merge(pd.DataFrame(submitters, columns=[patient_column]), data[[patient_column, category, site_column]], on=patient_column, how='left')
    newData.drop_duplicates(inplace=True)
    uniqueSites = data[site_column].unique()
    n = len(uniqueSites)
    listSet = []
    for v in values:
        listOrder = []
        for s in uniqueSites:
            listOrder += [len(newData[(newData[site_column] == s) & (newData[category] == v)].index)]
        listSet += [listOrder]
    gList = []
    for i in range(crossfolds):
        gList += [cp.Variable(n, boolean=True)]
    A = np.ones(n)
    constraints = [sum(gList) == A]
    error = 0
    for v in range(len(values)):
        for i in range(crossfolds):
            error += cp.square(cp.sum(crossfolds * cp.multiply(gList[i], listSet[v])) - sum(listSet[v]))
    prob = cp.Problem(cp.Minimize(error), constraints)
    prob.solve(solver='CPLEX', cplex_params={"timelimit": timelimit, "randomseed": randomseed})
    gSites = []
    for i in range(crossfolds):
        gSites += [[]]
    for i in range(n):
        for j in range(crossfolds):
            if gList[j].value[i] > 0.5:
                gSites[j] += [uniqueSites[i]]
    for i in range(crossfolds):
        str1 = "Crossfold " + str(i+1) + ": "
        j = 0
        for s in listSet:
            str1 = str1 + values[j] + " - " + str(int(np.dot(gList[i].value, s))) + " "
            j = j + 1
        str1 = str1 + " Sites: " + str(gSites[i])
        print(str1)
    for i in range(crossfolds):
        data.loc[data[site_column].isin(gSites[i]), target_column] = str(i+1)
    return data


if __name__ == "__main__":

    ori_label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_her2.csv')
    ori_label_df['site'] = ori_label_df['case_id'].apply(lambda x: x[5:7])

    presite_label_df = generate(ori_label_df, 'label', ['IDC', 'ILC'], crossfolds=5, target_column='CV', patient_column='case_id', site_column='site', timelimit=100, randomseed=0)

    presite_label_df.to_csv('../Dataset/TCGA-BRCA/tcga-brca_label_her2_presite5.csv', index=False)