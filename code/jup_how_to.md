---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Some useful commands in Jupyter

```python
# "!" allows you to run system commands
!conda env list
```

```python
!pip show pip
```

```python
# "!!" allows you to run system command and capture output as list
response = !! conda env list
```

```python
for r in response:
    print(r)
```

```python
%lsmagic
```

```python
%magic
```

```python

```
