use 5.010;
use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME               => 'PDL::ML::Preprocess',
    AUTHOR             => q{C.J. Collier <cjac@google.com>},
    VERSION_FROM       => 'lib/PDL/ML/Preprocess.pm',
    ABSTRACT_FROM      => 'lib/PDL/ML/Preprocess.pm',
    LICENSE            => 'perl_5',
    MIN_PERL_VERSION   => '5.010',
    CONFIGURE_REQUIRES => { 'ExtUtils::MakeMaker' => '0' },
    TEST_REQUIRES      => { 'Test::More' => '0.88' },
    PREREQ_PM          => { 'PDL' => '2.080', 'Carp' => '0' },
    META_MERGE         => {
        'meta-spec' => { version => 2 },
        resources   => {
            repository => {
                type => 'git',
                url  => 'https://github.com/PDLPorters/PDL-ML-Preprocess.git',
                web  => 'https://github.com/PDLPorters/PDL-ML-Preprocess',
            },
            bugtracker => {
                web => 'https://github.com/PDLPorters/PDL-ML-Preprocess/issues',
            },
        },
    },
    dist  => { COMPRESS => 'gzip -9f', SUFFIX => 'gz' },
    clean => { FILES => 'PDL-ML-Preprocess-*' },
);
