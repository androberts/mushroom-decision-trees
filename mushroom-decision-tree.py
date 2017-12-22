import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

'''
Build metadata dictionary, so values
can be decoded into descriptive labels
'''

metadata_csv = pd.read_csv('metadata.csv')
metadata = {}
for index, row in metadata_csv.iterrows():
    meta = {}
    metadata[row['field-name']] = meta
    for definition in row.attributes.split(' '):
        definition_split = definition.split('=')
        meta[definition_split[1]] = definition_split[0]


class Question:
    """
    A Question is used to evaluate a row against a
    given value, in order to partition the data.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        example_val = example[self.column]
        return example_val == self.value

    def __repr__(self):
        label_dict = metadata[data.columns[self.column]]
        label = label_dict[self.value]
        return "%s: %s?" % (header[self.column], str(label))


data = pd.read_csv('mushrooms.csv')
class_idx = 0  # The first column classifies the row as edible vs poisonous
header = data.columns


def unique_vals(rows, col):
    return set([r[col] for r in rows])


def class_counts(rows):
    counts = {}
    for r in rows:
        label = r[class_idx]
        counts[label] = (counts[label] + 1) if label in counts else 1
    return counts


def partition(rows, question):
    true_matches, false_matches = [], []
    for row in rows:
        if question.match(row):
            true_matches.append(row)
        else:
            false_matches.append(row)
    return true_matches, false_matches


def gini(partition):
    """
    Calculate Gini Impurity of the partition
    """
    counts = class_counts(partition)
    impurity = 1
    for lbl in counts:
        prob = counts[lbl] / len(partition)
        impurity -= prob ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p_left = float(len(left)) / (len(left) + len(right))
    p_right = 1 - p_left
    impurity_left = p_left * gini(left)
    impurity_right = p_right * gini(right)
    return current_uncertainty - impurity_left - impurity_right


def num_matches(list, class_val):
    return len([row for row in list if row[class_idx] == class_val])


def find_best_split(data):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(data.values)
    summary = []
    for i, col in enumerate(data.columns[1:]):
        values = data[col].unique()

        for val in values:
            question = Question(i + 1, val)
            true_rows, false_rows = partition(data.values, question)
            true_edible = num_matches(true_rows, 'e')
            true_poisonous = num_matches(true_rows, 'p')
            false_edible = num_matches(false_rows, 'e')
            false_poisonous = num_matches(false_rows, 'p')
            result = [question, 0, len(true_rows), gini(true_rows), true_edible, true_poisonous, len(false_rows),
                      gini(false_rows), false_edible, false_poisonous]
            summary.append(result)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)
            result[1] = gain

            if gain >= best_gain:
                best_gain, best_question = gain, question

    # Return summary as DataFrame with column headers
    summary_columns = ['Question',
                       'InfoGain',
                       'NumTrue',
                       'GiniTrue',
                       'EdibleTrue',
                       'PoisonousTrue',
                       'NumFalse',
                       'GiniFalse',
                       'EdibleFalse',
                       'PoisonousFalse']
    summary = pd.DataFrame(summary, columns=summary_columns).sort_values('InfoGain', ascending=False)
    return best_gain, best_question, summary


best_gain, best_question, summary = find_best_split(data)

print('best_gain: ', str(best_gain))
print('best_question: ', str(best_question))
print('num questions: ', len(summary))

'''
Render chart representing gini impurity of
'''

# Render charts
sum_head = summary.head(10)
sum_head = sum_head.reindex(index=sum_head.index[::-1])

ind = np.arange(len(sum_head))
width = 0.15

fig, ax = plt.subplots()

pois_true = ax.barh(ind, sum_head.GiniTrue, width, color='b')
pois_false = ax.barh(ind + width * 2, sum_head.GiniFalse, width, color='g')
ax.set_yticks(ind + width * 3 / 2)
labels = sum_head.Question.map(str)
ax.set_yticklabels(labels)


def label_rects(rects):
    for rect in rects:
        width = rect.get_width()
        ax.text(1.05 * rect.get_width(), rect.get_y() - rect.get_height() / 2.,
                '{0:.2f}'.format(width),
                ha='center', va='bottom', fontdict={'size': 8})


label_rects(pois_true)
label_rects(pois_false)
ax.legend((pois_true[0], pois_false[0]), ('"Yes"', '"No"'))
ax.set_title('Gini Impurity')
plt.subplots_adjust(left=.4, right=.9)
plt.show()
