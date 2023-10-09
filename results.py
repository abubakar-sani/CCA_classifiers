import json


def main():
    # Opening JSON file
    problems = ['binary', 'multi']
    metrics = ['accuracy', 'f1Score']
    estimators = ['LGBMClassifier', 'MLPClassifier', 'RandomForestClassifier', 'SVC', 'XGBClassifier']
    for estimator in estimators:
        for problem in problems:
            for metric in metrics:
                path = 'C:\\Users\\asata\\Documents\\KU\\CCA_classifiers\\results'
                filename = f'{estimator}_{problem}_{metric}'
                f = open(f'{path}\\{filename}.json')

                # returns JSON object as
                # a dictionary
                data = json.load(f)

                # Iterating through the json
                # list
                print(f'{filename}: {data}')

                # Closing file
                f.close()


if __name__ == '__main__':
    main()
