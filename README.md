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

# Logging

The Elastic stack (formerly ELK stack) is a common option for production level logging. It combines the features of Elasticsearch (distributed search engine), Logstash (ingestion pipeline) and Kibana (customizable visualization). We could also simply upload our logs to a cloud blog storage (ex. S3, Google Cloud Storage, etc.).

# Best practices for API serving

When designing our API, there are some best practices to follow:

- URI paths, messages, etc. should be as explicit as possible. Avoid using cryptic resource names, etc.
- Use nouns, instead of verbs, for naming resources. The request method already accounts for the verb (✅ GET /users not ❌ GET /get_users).
- Plural nouns (✅ GET /users/{userId} not ❌ GET /user/{userID}).
- Use dashes in URIs for resources and path parameters but use underscores for query parameters (GET /nlp-models/?find_desc=bert).
- Return appropriate HTTP and informative messages to the user.

# Serving with RESTfull API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```
