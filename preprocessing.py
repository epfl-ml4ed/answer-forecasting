import pandas as pd
import numpy as np
import math
import os

def preprocess_lernnavi_qna(basepath="./data/lernnavi", verbose=True):
    questions_and_masteries = pd.read_pickle(f"{basepath}/qna/MULTIPLE_CHOICE_german_with_mastery.pkl").reset_index()
    
    # take first train_size samples for each user ordered by start_time
    train_size, validation_size, test_size = 0.7, 0.2, 0.1
    assert math.isclose(train_size + validation_size + test_size, 1)

    train_set = (
        questions_and_masteries
            .groupby("user_id")
            .apply(lambda x: x.sort_values("start_time").iloc[:int(len(x) * train_size)])
            .reset_index(drop=True)
    )
    validation_set = (
        questions_and_masteries
            .groupby("user_id")
            .apply(lambda x: x.sort_values("start_time").iloc[int(len(x) * train_size):int(len(x) * (train_size + validation_size))])
            .reset_index(drop=True)
    )
    test_set = (
        questions_and_masteries
            .groupby("user_id")
            .apply(lambda x: x.sort_values("start_time").iloc[int(len(x) * (train_size + validation_size)):])
            .reset_index(drop=True)
    )

    if verbose:
        print("Performed train/validation/test split:")
        print(f"Training set size: {len(train_set)}")
        print(f"Validation set size: {len(validation_set)}")
        print(f"Test set size: {len(test_set)}")

    modes = zip(
        ["train", "validation", "test"],
        [train_set, validation_set, test_set],
    )

    for mode, data in modes:
        os.makedirs(f"{basepath}/qna/{mode}", exist_ok=True)
        data.to_pickle(f"{basepath}/qna/{mode}/qna.pkl")


def preprocess_lernnavi_data(basepath="./data/lernnavi", verbose=True):
    # load data
    users = pd.read_csv(f"{basepath}/user_data.csv")
    topics = pd.read_csv(f"{basepath}/mastery_per_topic.csv")

    # filter out eventually inconsistent data
    common_users = set.intersection(set(users["user_id"]), set(topics["user_id"]))
    if verbose:
        print(f"Removed {len(set.union(set(users['user_id']), set(topics['user_id']))) - len(common_users)} users from data.")
    
    users = users[users["user_id"].isin(common_users)]
    topics = topics[topics["user_id"].isin(common_users)]

    # adjust index and column names
    users = (
        users
            .set_index("user_id")
            .sort_index()
            .rename(columns={
                "0": "mastery_mean",
                "final_w": "normalized_score_weighted_by_difficulty",
                "final_nw": "normalized_score",
            })
            .fillna(0)
    )
    topics = (
        topics
            .set_index("user_id")
            .sort_index()
            .fillna(0)
    )

    # split data into train, validation and test set
    validation_size, test_size = 0.2, 0.1
    valtest_user_ids = np.random.choice(users.index, int((validation_size + test_size) * len(users)), replace=False)
    validation_user_ids = valtest_user_ids[:int(validation_size * len(users))]
    test_user_ids = valtest_user_ids[int(validation_size * len(users)):]

    if verbose:
        print("Performed train/validation/test split:")
        print(f"Training set size: {len(users) - len(validation_user_ids) - len(test_user_ids)}")
        print(f"Validation set size: {len(validation_user_ids)}")
        print(f"Test set size: {len(test_user_ids)}")

    test_users = users.loc[test_user_ids]
    validation_users = users.loc[validation_user_ids]
    training_users = users.drop(np.concatenate((validation_user_ids, test_user_ids)))

    test_topics = topics.loc[test_user_ids]
    validation_topics = topics.loc[validation_user_ids]
    training_topics = topics.drop(np.concatenate((validation_user_ids, test_user_ids)))

    modes = zip(
        ["train", "validation", "test"],
        [training_users, validation_users, test_users],
        [training_topics, validation_topics, test_topics],
    )

    for mode, users_df, topics_df in modes:
        os.makedirs(f"{basepath}/{mode}", exist_ok=True)
        users_df.to_csv(f"{basepath}/{mode}/users.csv")
        topics_df.to_csv(f"{basepath}/{mode}/topics.csv")

if __name__ == "__main__":
    # preprocess_lernnavi_data()
    preprocess_lernnavi_qna()