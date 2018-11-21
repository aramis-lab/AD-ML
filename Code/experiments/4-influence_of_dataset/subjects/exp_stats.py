

def unique_file_for_iteration(input_path, database_name, image_type, classification, pvc, task, n_iteration, classifier,
                              variables):
    import os
    import numpy as np
    import pandas as pd

    balanced_accuracy = []
    # path = os.path.join(input_path, database_name, 'outputs', image_type, classification)
    path = os.path.join(input_path, image_type, classification)
    if image_type == 'fdg':
        path = os.path.join(path, 'pvc-' + pvc)
    for i in xrange(n_iteration):
        if os.path.isfile(os.path.join(path, variables, classifier, task, 'iteration-' + str(i), 'results.tsv')):
            balanced_accuracy.append((pd.io.parsers.read_csv(
                os.path.join(path, variables, classifier, task, 'iteration-' + str(i), 'results.tsv'),
                sep='\t')).balanced_accuracy[0])

    print (np.mean(balanced_accuracy), np.std((np.array(balanced_accuracy))))


import pandas as pd
import numpy as np


def get_bal_std(input_file):
    df = pd.io.parsers.read_csv(input_file, sep='\t')
    print df.balanced_accuracy.mean()
    print df.balanced_accuracy.std()

