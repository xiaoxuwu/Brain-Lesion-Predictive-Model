# manifest constants
TARGET_COL = 622
FEATURE = 0
TARGET = 1

def get_csv(path):
    data = pd.read_csv(filepath_or_buffer=path, header=None)
    # all train data
    X = data.iloc[:,3:-1]
    # all test data
    Y = data.iloc[:, -1:][TARGET_COL]
    return (X, Y, data)

def part_list(lst, n):
    """
        part_list: Partition lst balanced parts
        in:
            lst - list that needs to be partitioned
            n - integer number of partitions
        out:
            partitioned list
    """
    parts, rest = divmod(len(lst), n)
    lstiter = iter(lst)
    for j in xrange(n):
        plen = len(lst)/n + (1 if rest > 0 else 0)
        rest -= 1
        yield list(itertools.islice(lstiter, plen))

def build_group_df(data, patients):
    """
        build_group_df: helper for build_cross_validation_sets
        in:
            data - RAW data
            patients - list of patient ids
        out:
            df with concatenated pixel data relevant to each patient in patients
    """
    return pd.concat([data[data[0] == patient] for patient in patients], ignore_index=True)

def build_cross_validation_sets(data, k):
    """
        build_cross_validation_sets: helper for cross_validate
        in:
            data: RAW data
            k - desire number of groups
        out:
            list of tuples: (feature_df, target_series)
    """
    # manifest constants, get unique patients, and random shuffle
    unique_patients = data[0].unique().tolist()
    random.shuffle(unique_patients)

    #create k groups
    k_groups = list(part_list(unique_patients, k))

    # [df1, df2, df3,...dfi,...dfk] with each dfi repersenting the ith group in k total groups
    k_df = [build_group_df(data, group) for group in k_groups]
    # build a list [(features, target) for each df]
    k_df_split = [(data.iloc[:,4:-1], data.iloc[:, -1:][TARGET_COL]) for data in k_df]

    return k_df_split

def cross_validate(model, data, k = 5):
    """
        cross_validate: performs cross validation
        in:
            model - input model
            data - RAW data
            k - desired number of groups
        out:
            (mean of scores, list of scores)
    """
    # manifest constants
    score_list = []

    # get split data
    k_df_split = build_cross_validation_sets(data, k)

    for (i, (X, y)) in enumerate(k_df_split):
        # get all dfs not k
        non_kth_group = k_df_split[:]
        del non_kth_group[i]

        # build x and y train data
        X_train = pd.concat([data[FEATURE] for data in non_kth_group])
        y_train = pd.concat([data[TARGET] for data in non_kth_group])

        # build x and y test data
        X_test = X
        y_test = y

        # train model on non_kth_group
        model.fit(X_train, y_train)

        # test model on kth group
        score = model.score(X_test, y_test)

        # add score to score list
        score_list.append(score)

    return (np.mean(score_list), score_list)
