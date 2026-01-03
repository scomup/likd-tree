# Uploading likd-tree to PyPI

```bash
rm -rf build dist *.egg-info
python3 -m build --sdist
python3 -m twine upload  dist/likd-tree-1.0.0.tar.gz
```