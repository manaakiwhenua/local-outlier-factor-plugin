[![manaakiwhenua-standards](https://github.com/manaakiwhenua/local-outlier-factor-plugin/workflows/manaakiwhenua-standards/badge.svg)](https://github.com/manaakiwhenua/manaakiwhenua-standards)

# local-outlier-factor-plugin

## Building for release

Requires wheel.

`python setup.py sdist bdist_wheel`

This can be included in a requirements.txt as: `git+https://github.com/manaakiwhenua/local-outlier-factor-plugin.git@master`

`master` branch is for release, changes should be proposed in a separate branch and a PR submitted for merging into master, including rebuilding the source distributions.

(TODO: replace with a Github Action)
