import numpy as np
import math

from src.mine import mine_fa


class fsnid_selection:

    def __init__(
        self,
        features,
        targets,
        num_iterations=10000,
        mi_ordering_bool=True
    ):

        # X -> network traffic features
        self.features = features

        # Y -> attack/normal labels
        self.targets = targets

        # How long mine_fa should train
        self.num_iterations = num_iterations

        # Whether features should first be ordered
        # according to their individual information
        self.mi_ordering_bool = mi_ordering_bool

        # We calculate the random-noise threshold once
        # when FSNID starts
        self.nm_upper_bound = self.null_model()


    def mine(self, features, targets):
        """
        Estimate how informative 'features'
        are about 'targets'.
        """

        miner = mine_fa(
            p_dis=features,
            q_dis=targets,
            num_iterations=self.num_iterations
        )

        return miner.run()
    def null_model(self):

        # Create completely random fake features
        random_features = np.random.random(
            self.targets.shape
        )

        # Measure how much "information"
        # random noise appears to have about Y
        nm_arr = self.mine(
            random_features,
            self.targets
        )

        # Take an upper confidence bound
        nm_upper_bound = (
            nm_arr.mean(axis=0)[-1]
            + 2 * (
                nm_arr.std(axis=0)[-1]
                / math.sqrt(3)
            )
        )

        return nm_upper_bound
    def run_main(self):

        # Initially keep ALL features
        feats = list(
            range(self.features.shape[1])
        )

        # Decide the order in which
        # features will be checked
        if self.mi_ordering_bool:
            iterable_feats = self.mi_ordering()
        else:
            iterable_feats = range(
                self.features.shape[1]
            )


        # Check every feature one by one
        for feat in iterable_feats:

            # Current feature set without 'feat'
            temp_indexes = [
                item for item in feats
                if item != feat
            ]


            if len(feats) > 1:

                # Information using current features
                info_with = self.mine(
                    self.features[:, feats],
                    self.targets
                )

                # Information after removing this feature
                info_without = self.mine(
                    self.features[:, temp_indexes],
                    self.targets
                )

                # Φ = information lost when feature is removed
                arr = info_with - info_without

            else:

                # Only one feature remains
                arr = self.mine(
                    self.features[:, feats],
                    self.targets
                )


            # Lower confidence bound of Φ
            lower_bound = (
                arr.mean(axis=0)[-1]
                - 2 * (
                    arr.std(axis=0)[-1]
                    / math.sqrt(3)
                )
            )


            # If feature is not better than random noise,
            # remove it
            if lower_bound < self.nm_upper_bound:

                feats = temp_indexes

                print(
                    f"Feature {feat} excluded | "
                    f"Phi lower bound = {lower_bound:.6f} | "
                    f"Null threshold = {self.nm_upper_bound:.6f}"
                )

            else:

                print(
                    f"Feature {feat} included | "
                    f"Phi lower bound = {lower_bound:.6f} | "
                    f"Null threshold = {self.nm_upper_bound:.6f}"
                )


        return feats
    def mi_ordering(self):

        # Store individual information score
        # of every feature
        mis = []

        for feat in range(self.features.shape[1]):

            # Take ONLY one feature
            single_feature = self.features[:, feat]

            # How informative is this feature alone about Y?
            mi_result = self.mine(
                single_feature,
                self.targets
            )

            # Take final information estimate
            mi_score = mi_result.mean(axis=0)[-1]

            mis.append(mi_score)


        # Sort feature indices from
        # lowest information → highest information
        sorted_indices = sorted(
            range(len(mis)),
            key=lambda i: mis[i]
        )

        return sorted_indices