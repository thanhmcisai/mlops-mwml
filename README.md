### Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```

- 'Config' directory: where we can store components that will be required for our application

# Principles

## Wrapping functionality into functions

It's better to wrap them as a separate function because we may want to:

- repeat this functionality in other parts of the project or in other projects.
- test that these tags are actually being replaced properly.

```python
oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
```

──── compared to ────

```python
def replace_oos_tags(df, tags_dict):
    """Replace out of scope (oos) tags."""
    oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
    df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
    return df
```

## Composing generalized functions

```python
# Specific
def replace_oos_tags(df, tags_dict):
    """Replace out of scope (oos) tags."""
    oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
    df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
    return df
```

──── compared to ────

```python
# Generalized
def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Replace out of scope (oos) labels."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df
```

##
