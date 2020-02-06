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
#keyboard shortcuts:
#  <ESC> : switch to command mode
#      a : insert cell above current location
#      b : insert cell above current location
#      x : delete current cell
#      z : undo cell operation
#      c : copy cell
#      p : paste cell below currently selected cell
#<Ctrl> s: Save
```

```python
#keyboard shortcuts:
#        <ENTER> : switch to edit mode
#          <TAB> : code completion or indent
#  <Shift> <TAB> : tooltip
#        <Ctrl> z: undo
# <Shift> <Enter>: run cell, select below
#        <Ctrl> s: Save
#
# on macOS, use <CMD> instead of <Ctrl>
```

```python
mystr = 'Test'

```

```python
# now you can use tab, shift tab
# start typing variable name and hit <TAB>
# once have variable or function, can hit <SHIFT> and <TAB> to get documentation
my
```

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
import time
```

```python
time.sleep(1)
```

```python
%timeit time.sleep(1)
```

```python
%timeit

for i in range(1, 5):
    time.sleep(1)
    
print('Done')
```

```python
%%timeit

for i in range(1, 5):
    time.sleep(1)
    
print('Done')
```

```python
# can either request docstring with "?"
# or use <SHIFT><TAB>
%timeit?
```

```python
#%store: Pass variables between notebooks.
%store mystr
del mystr
print(mystr)
```

```python
# retrieve variable in this or any other notebook running on same Jupyter instance
%store -r mystr
print(mystr)
```

```python

```

```python
%who: List all variables of global scope.
```

```python
%run sample.py
```

```python
%load sample.py
```
