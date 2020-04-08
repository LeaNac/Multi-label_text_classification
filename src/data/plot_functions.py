import matplotlib.pyplot as plt
import seaborn as sns


def plot_length_comments_hist(df):
    lens = df.comment_text.str.len()
    print(f'Mean length :{lens.mean()}, Std :{lens.std()}, Max:{lens.max()}')
    return lens.hist()


def plot_nb_of_labels_per_comments(df):
    rowsums = df.iloc[:, 2:].sum(axis=1)
    x = rowsums.value_counts().sort_index()

    # plot
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x.index, x.values, alpha=0.8)
    plt.title("Multiple tags per comment")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('# of tags ', fontsize=12)

    # adding the text labels
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

    plt.show()
