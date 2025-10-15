git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config filter.strip-notebook-output.smudge 'cat'
git config filter.strip-notebook-output.required true