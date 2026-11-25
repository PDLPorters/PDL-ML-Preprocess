use strict;
use warnings;
use Test::More;
use PDL;
use PDL::ML::Preprocess qw(
    train_test_split
    standard_scale
    minmax_scale
    l2_normalize
    one_hot_encode
);

subtest 'UCI E. coli protein localization preprocessing tutorial' => sub {
    # 6 samples x 4 signal-peptide features (mcg, gvh, lip, aac)
    my $X = pdl(double, [
        [0.49, 0.29, 0.48, 0.56],
        [0.07, 0.40, 0.48, 0.54],
        [0.56, 0.40, 0.48, 0.49],
        [0.83, 0.37, 0.48, 0.61],
        [0.63, 0.49, 0.48, 0.50],
        [0.24, 0.35, 0.48, 0.52],
    ]);
    my $y = pdl(long, [0, 0, 1, 1, 2, 2]);

    my ($X_tr, $X_te, $y_tr, $y_te) = train_test_split($X, $y, test_size => 0.33, seed => 42);
    is($X_tr->dim(-1) + $X_te->dim(-1), 6, 'split preserves total sample count');
    is($y_tr->nelem, $X_tr->dim(-1), 'train labels match train samples');
    is($y_te->nelem, $X_te->dim(-1), 'test labels match test samples');

    my ($scaled) = standard_scale($X);
    is_deeply([$scaled->dims], [$X->dims], 'standard_scale preserves shape');

    my ($mm) = minmax_scale($X);
    ok($mm->min >= -1e-9 && $mm->max <= 1 + 1e-9, 'minmax_scale bounds in [0, 1]');

    my $oh = one_hot_encode($y, 3);
    is($oh->dim(0), 3, 'one_hot_encode has 3 classes');
    is($oh->dim(1), 6, 'one_hot_encode has 6 samples');
};

subtest 'disconfirmation: mismatched X/y sample counts' => sub {
    my $X = pdl(double, [[1, 2], [3, 4], [5, 6]]);
    my $y = pdl(long, [0, 1]);
    my $ok = eval { train_test_split($X, $y); 1 };
    ok(!$ok, 'train_test_split croaks on mismatched X and y sample counts');
    like($@, qr/matching sample counts/i, 'diagnostic error message');
};

done_testing();
