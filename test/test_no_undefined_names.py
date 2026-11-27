"""A name a script uses but never defines only fails when that line runs. train.py is a script, so
no test imports it: `teacher_config` was set inside set_config_defaults() and read in __main__, and
every model stopped at that line, on the first launch."""
import ast
import builtins
import pathlib
import symtable

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = [ROOT / 'train.py']
for directory in ('models', 'utils', 'tools'):
    FILES += sorted((ROOT / directory).glob('*.py'))

# Known, outside the code written here: models/cosmos_predict2_modeling.py is NVIDIA's, and its
# 'transformer_engine' attention backend names a class it never imports (that backend is not used).
KNOWN = {('models/cosmos_predict2_modeling.py', 'DotProductAttention')}


def undefined_names(source):
    tree = ast.parse(source)
    if any(isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names) for node in ast.walk(tree)):
        return set()
    top = symtable.symtable(source, '<module>', 'exec')
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    defined |= set(dir(builtins)) | {'__file__', '__name__', '__doc__', '__spec__', '__builtins__'}
    missing = set()

    def declared(table):
        # `global x` in a function that assigns it (a worker initializer) defines x for the module
        for symbol in table.get_symbols():
            if symbol.is_declared_global() and symbol.is_assigned():
                defined.add(symbol.get_name())
        for child in table.get_children():
            declared(child)

    declared(top)

    def walk(table):
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if not symbol.is_referenced() or name in defined:
                continue
            if table is top or symbol.is_global():
                missing.add(name)
        for child in table.get_children():
            walk(child)

    walk(top)
    return missing


@pytest.mark.parametrize('path', FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_name_is_defined(path):
    missing = undefined_names(path.read_text())
    missing = {name for name in missing if (str(path.relative_to(ROOT)), name) not in KNOWN}
    assert not missing, f'{path.name} uses names it never defines: {sorted(missing)}'
