# Copyright 2026 Google LLC
use strict;
use warnings;
use PDL;
use PDL::ML::Preprocess qw(
    train_test_split
    standard_scale
    minmax_scale
    l2_normalize
    one_hot_encode
);

# 1. Synthetic dataset: 3 features x 10 samples, 3 classes
my $X = sequence(double, 3, 10) + 1.0;
my $y = long([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]);

# 2. Reproducible 80/20 train/test split (sample axis is dim -1)
my ($X_tr, $X_te, $y_tr, $y_te) = train_test_split(
    $X, $y, test_size => 0.2, shuffle => 1, seed => 42,
);
printf "Split: train=%d samples, test=%d samples\n", $X_tr->dim(-1), $X_te->dim(-1);

# 3. Leak-free standard scaling: fit on train, apply to test
my ($X_tr_std, $mean, $std) = standard_scale($X_tr);
my $X_te_std = standard_scale($X_te, mean => $mean, std => $std);
print 'Train mean: ', $mean, "\n";
print 'Test standardized:', "\n", $X_te_std, "\n";

# 4. Leak-free min-max scaling: fit on train, apply to test
my ($X_tr_mm, $min, $max) = minmax_scale($X_tr);
my $X_te_mm = minmax_scale($X_te, min => $min, max => $max);
print 'Test min-max scaled:', "\n", $X_te_mm, "\n";

# 5. L2-normalize feature vectors (unit norm along dim 0)
my $X_unit = l2_normalize($X_tr);
my $norms  = sqrt(($X_unit * $X_unit)->sumover);
print 'Unit norms: ', $norms, "\n";

# 6. One-hot encode integer class labels into (num_classes, n_samples)
my $Y_tr_oh = one_hot_encode($y_tr, 3);
print 'One-hot labels (3 x ', $y_tr->nelem, '):', "\n", $Y_tr_oh, "\n";
