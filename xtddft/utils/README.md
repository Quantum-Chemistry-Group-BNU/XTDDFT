# Content of this document
### `atom.py`
Some Cartesian coordinate of test molecules

### `unit.py`
Some Unit conversion

### `utils.py`
1. Transform spin orbital basis to spin tensor basis
2. Transform spin tensor basis to spin orbital basis
3. According to `mf`, return number of doubly occupied orbitals (nc),
number of singly occupied orbitals (no) and number of virtual orbitals (nv)
4. Transform pyscf $A$ matrix order to CV(0), OV(0), CO(0), CV(1). Note, only for open shell sTDA method
5. Two memory statistics tools, one to count the maximum memory usage of a program segment, 
and another to count the maximum memory usage at a specific moment.