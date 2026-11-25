# PDL::ML::Preprocess

Machine learning tabular and vector preprocessing utilities (`train_test_split`, `standard_scale`, `minmax_scale`, `l2_normalize`, `one_hot_encode`) operating directly on [PDL](https://pdl.perl.org/) ndarrays.

## Installation

```bash
cpanm .
```

Or via ExtUtils::MakeMaker:

```bash
perl Makefile.PL
make
make test
make install
```

## Quickstart

```perl
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

my $X = sequence(double, 3, 10) + 1.0;
my $y = long([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]);

my ($X_tr, $X_te, $y_tr, $y_te) = train_test_split(
    $X, $y, test_size => 0.2, shuffle => 1, seed => 42,
);

# Fit on training split, apply to test split
my ($X_tr_std, $mean, $std) = standard_scale($X_tr);
my $X_te_std = standard_scale($X_te, mean => $mean, std => $std);

my $Y_tr_oh = one_hot_encode($y_tr, 3);
```

## Documentation & Examples

- Runnable tutorial: `perl examples/quickstart.pl` (see [`examples/quickstart.pl`](examples/quickstart.pl))
- Full POD documentation: `perldoc PDL::ML::Preprocess` (or [`lib/PDL/ML/Preprocess.pm`](lib/PDL/ML/Preprocess.pm))

## License

Copyright 2026 Google LLC. Licensed under the the same terms as Perl itself.
