import torch


class RandomSampler(object):
    """Random sampler.
    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
    """

    def __init__(self, num, pos_fraction, neg_pos_ub=-1):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub

    def sample(self, assign_gts_inds):
        num_expected_pos = int(self.num * self.pos_fraction)

        pos_inds = self._sample_pos(assign_gts_inds, num_expected_pos)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()

        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(assign_gts_inds, num_expected_neg)
        neg_inds = neg_inds.unique()
        return pos_inds, neg_inds

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.
        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.
        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.
        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device()
            )
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_gts_inds, num_expected):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_gts_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_gts_inds, num_expected):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_gts_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
