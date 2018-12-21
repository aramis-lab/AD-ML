
def classification_results(results_path, database_name, image_type, pvc, classification_type, features, classifiers, tasks):
    """
    Aim of this method is to load in a single tsv file all the results for a classification, so it can be easier the analysis of the results through filters in excel.
    It's a guide to create the tables for the analysis of the results

    :param results_path: where classification results are saved
    :param database_name: ADNI,AIBL,OASIS
    :param image_type: fdg, T1
    :param pvc: rbv, None
    :param classification_type: region_based, voxel_based
    :param features: list of features
    :param classifiers: list of classifiers
    :param tasks: list of tasks

    :return: DataFrame containing results for each classification

    """
    import os
    import pandas as pd

    classif_path = os.path.join(results_path, database_name, 'outputs', image_type, classification_type)
    if image_type == 'fdg':
        classif_path = os.path.join(classif_path, 'pvc-' + pvc)

    accuracy = []
    auc = []
    balanced_accuracy = []
    npv = []
    ppv = []
    sensitivity = []
    specificity = []
    atl = []
    classifier = []
    task_clas = []
    smooth = []

    for feat in features:
        for classif in classifiers:
            for task in tasks:
                if os.path.exists(os.path.join(classif_path, feat, classif, task)):
                    task_path = os.path.join(classif_path, feat, classif, task)
                    print task_path
                    if os.path.isfile(os.path.join(task_path, 'mean_results.tsv')):
                        mean_result = pd.io.parsers.read_csv(os.path.join(task_path, 'mean_results.tsv'), sep='\t')
                        accuracy.append(mean_result.accuracy[0])
                        auc.append(mean_result.auc[0])
                        balanced_accuracy.append(mean_result.balanced_accuracy[0])
                        npv.append(mean_result.npv[0])
                        ppv.append(mean_result.ppv[0])
                        sensitivity.append(mean_result.sensitivity[0])
                        specificity.append(mean_result.specificity[0])
                    else:
                        accuracy.append('')
                        auc.append('')
                        balanced_accuracy.append('')
                        npv.append('')
                        ppv.append('')
                        sensitivity.append('')
                        specificity.append('')

                    if classification_type == 'voxel_based':
                        smooth.append(feat)
                    else:
                        atl.append(feat)

                    classifier.append(classif)
                    task_clas.append(task)
    # print classif_path
    #
    # print len(accuracy)
    # print len(auc)
    # print len(balanced_accuracy)
    # print len(npv)
    # print len(ppv)
    # print len(sensitivity)
    # print len(specificity)
    # print len(atl)
    # print len(classifier)
    # print len(task_clas)
    # print len(smooth)

    if classification_type == 'region_based':
        # print 'REGION BASED!!'
        stats = pd.DataFrame({'atlas': atl,
                              'classifier': classifier,
                              'task': task_clas,
                              'accuracy': accuracy,
                              'auc': auc,
                              'balanced_accuracy': balanced_accuracy,
                              'npv': npv,
                              'ppv': ppv,
                              'sensitivity': sensitivity,
                              'specificity': specificity
                              })
    else:
        stats = pd.DataFrame({'smoothing': smooth,
                              'classifier': classifier,
                              'task': task_clas,
                              'accuracy': accuracy,
                              'auc': auc,
                              'balanced_accuracy': balanced_accuracy,
                              'npv': npv,
                              'ppv': ppv,
                              'sensitivity': sensitivity,
                              'specificity': specificity
                              })
    return stats


def create_table(results_path, databases, image_types, classification_types, pvc_applied, table_path):

    import os
    import pandas as pd

    all_dataframes = []

    for database in databases:
        for image_type in image_types:
            for classification_type in classification_types:
                for pvc in pvc_applied:
                    if (image_type == 'T1') and (pvc != 'None'):
                        continue

                    df = classification_results(results_path, database, image_type, pvc, classification_type,
                                                classification_types[classification_type]['features'],
                                                classification_types[classification_type]['classifiers'],
                                                databases[database])
                    df['database'] = database
                    df['features'] = classification_type
                    df['modality'] = image_type
                    if image_type == 'fdg':
                        df['pvc'] = pvc

                    all_dataframes.append(df)

    all_results = pd.concat(all_dataframes)
    # all_results.to_csv(os.path.join(table_path, 'table_all_results_with_blanks.tsv'), sep='\t', index=False,
    #                    encoding='utf8', columns=['database', 'modality', 'pvc', 'features', 'atlas', 'smoothing',
    #                                              'classifier', 'task', 'balanced_accuracy', 'auc', 'sensitivity',
    #                                              'specificity', 'accuracy', 'npv', 'ppv'])

    results = (all_results[~(all_results.accuracy == '')])

    results.to_csv(os.path.join(table_path, 'table_all_results.tsv'), sep='\t', index=False, encoding='utf8',
                   columns=['database', 'modality', 'pvc', 'features', 'atlas', 'smoothing', 'classifier', 'task',
                            'balanced_accuracy', 'auc', 'sensitivity', 'specificity', 'accuracy', 'npv', 'ppv'])

    return results


# Running the script

results_path = '/Volumes/aramis-projects/simona.bottani/ADML_paper'
databases = {'ADNI': ['CN_vs_AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'CN-_vs_AD+', 'CN-_vs_MCI+','CN-_vs_pMCI+', 'MCI-_vs_MCI+', 'sMCI_vs_pMCI', 'sMCI+_vs_pMCI+'],
             'AIBL': ['CN_vs_AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI'],
             'OASIS': ['CN_vs_AD']}
image_types = ['T1', 'fdg']
classification_types = {'region_based': {'classifiers': ['linear_svm', 'logistic_regression', 'random_forest'],
                                         'features': ['atlas-AAL2', 'atlas-Neuromorphometrics', 'atlas-Hammers', 'atlas-LPBA40', 'atlas-AICHA']
                                         },
                        'voxel_based': {'classifiers': ['linear_svm'],
                                        'features': ['smooothing-0', 'smooothing-4', 'smooothing-8', 'smooothing-12']
                                        }
                        }
pvc_applied = ['rbv', 'None']
table_path = '/Users/jorge.samper/ownCloud/ADMLpaper/Classification/Results'

df = create_table(results_path, databases, image_types, classification_types, pvc_applied, table_path)
