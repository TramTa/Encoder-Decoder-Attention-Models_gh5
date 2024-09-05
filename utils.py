# utils.py

from typing import List
import numpy as np

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def maybe_add_feature(feats: List[int], feature_indexer: Indexer, add_to_indexer: bool, feat: str):
    """
    :param feats: list[int] features that we've built so far
    :param feature_indexer: Indexer object to apply
    :param add_to_indexer: True if we should expand the Indexer, false otherwise. If false, we discard feat if it isn't
    in the indexer
    :param feat: new feature to index and potentially add
    :return:
    """
    if add_to_indexer:
        feats.append(feature_indexer.add_and_get_index(feat))
    else:
        feat_idx = feature_indexer.index_of(feat)
        if feat_idx != -1:
            feats.append(feat_idx)


def score_indexed_features(feats, weights: np.ndarray):
    """
    Computes the dot product over a list of features (i.e., a sparse feature vector)
    and a weight vector (numpy array)
    :param feats: List[int] or numpy array of int features
    :param weights: numpy array
    :return: the score
    """
    score = 0.0
    for feat in feats:
        score += weights[feat]
    return score


