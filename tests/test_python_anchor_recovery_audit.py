"""Independent conservative audit of full-function formatting recovery."""
import hashlib
import time

import pytest

from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.python_anchor import recover_function_anchor
from patchrook_agent.tools import ToolConfig,Workspace

SOURCE='def target(value):\n    return value - 1\n'
OLD='    def target( value ):\n      return value-1\n'
NEW='    def target( value ):\n      return value+1\n'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a,**k):pytest.fail('Anchor audit must not execute models or HTTP')
    monkeypatch.setattr('httpx.Client',forbidden)
    monkeypatch.setattr('httpx.get',forbidden)
    monkeypatch.setattr('httpx.post',forbidden)


def recover(source,old=OLD,new=NEW,ranges=None):
    return recover_function_anchor(source,old,new,[(0,len(source))] if ranges is None else ranges)


def editor(root,source=SOURCE,*,enabled=True):
    (root/'subject.py').write_text(source)
    tools=EvidenceTools(Workspace(root,ToolConfig(validate_python_edits=False)),recover_python_function_anchors=enabled)
    return tools


def view(tools,first=1,last=None):
    text=(tools.workspace.root/'subject.py').read_text();lines=text.splitlines(keepends=True)
    last=len(lines) if last is None else last
    return {'path':'subject.py','start_line':first,'end_line':last,
        'content':''.join(lines[first-1:last]),'source_sha256':hashlib.sha256(text.encode()).hexdigest()}


def apply(tools,edits,sources=None):
    return tools.execute('apply_edits',{'edits':edits},deadline=time.monotonic()+10,visible_sources=sources)


def change(old=OLD,new=NEW):return {'path':'subject.py','old_text':old,'new_text':new}


@pytest.mark.parametrize('current,model_return',[
    ('1_000','1000'),('0x10','16'),('1.0','1.'),('"value"',"'value'"),
    ('"é"','"\\u00e9"'),("'a' 'b'","'ab'"),('(value)','value')])
def test_ast_equivalence_never_normalizes_literal_or_operator_spelling(current,model_return):
    source='def target(value):\n    return '+current+'\n'
    old='  def target( value ):\n    return '+model_return+'\n'
    assert recover(source,old,NEW) is None


@pytest.mark.parametrize('body',[
    'class Owner:\n    def target(value):\n        return value - 1\n\n',
    'def outer():\n    def target(value):\n        return value - 1\n    return target\n\n'])
def test_matching_method_or_inner_function_makes_top_level_anchor_ambiguous(body):
    assert recover(body+SOURCE) is None


def test_unique_inner_function_is_not_promoted_to_top_level():
    source='class Owner:\n    def target(value):\n        return value - 1\n'
    assert recover(source) is None


def test_same_named_later_definition_prevents_dead_scope_rewrite():
    assert recover(SOURCE+'\ndef target(value):\n    return value + 2\n') is None


@pytest.mark.parametrize('source,old,new',[
    ('@decorate\n'+SOURCE,OLD,NEW),
    ('async def target(value):\n    return value - 1\n',OLD,NEW),
    (SOURCE.replace('value - 1','value - 1 # preserve'),OLD,NEW),
    (SOURCE,OLD,NEW+'    # newly proposed comment\n'),
    ('def target(value):\n    return """line1\nline2"""\n',
     'def target( value ):\n  return """line1\nline2"""\n','def target(value):\n  return "updated"\n'),
    ('def target(value):\n    return f"{value}"\n',
     'def target( value ):\n  return f"{value}"\n','def target(value):\n  return str(value)\n'),
    ('def target(value):\n    return f"""text\n{value}"""\n',
     'def target( value ):\n  return f"""text\n{value}"""\n','def target(value):\n  return str(value)\n')])
def test_decorators_async_comments_and_interpolated_or_multiline_strings_skip(source,old,new):
    assert recover(source,old,new) is None


@pytest.mark.parametrize('source,old,new',[
    ('def target(value, *, scale=1):\n    return value * scale\n',
     'def target( value, *, scale=1 ):\n  return value*scale\n',
     'def target(value, *, scale=2):\n  return value*scale\n'),
    ('def target(value) -> int:\n    return value - 1\n',
     'def target( value ) -> int:\n  return value-1\n',
     'def target(value) -> str:\n  return str(value)\n'),
    ('def target[T: int](value: T) -> T:\n    return value\n',
     'def target[T: int]( value: T ) -> T:\n  return value\n',
     'def target[T: str](value: T) -> T:\n  return value\n')])
def test_new_public_signature_includes_keyword_defaults_annotations_and_type_parameters(source,old,new):
    assert recover(source,old,new) is None


def test_utf8_offsets_preserve_adjacent_module_code(tmp_path):
    source='def target(value):\n    return "é🧩é"\nOUTSIDE = "keep"\n'
    old='    def target( value ):\n      return "é🧩é"\n'
    new='    def target( value ):\n      return "updated"\n'
    tools=editor(tmp_path,source)
    result=apply(tools,[change(old,new)],[view(tools)])
    assert result['success']
    actual=(tmp_path/'subject.py').read_text()
    assert 'return "updated"' in actual and 'OUTSIDE = "keep"\n' in actual
    compile(actual,'subject.py','exec')


def test_trailing_nonformat_semicolon_is_not_lost_from_scope_proof():
    source='def target(value): return value - 1;\nOUTSIDE = 9\n'
    assert recover(source) is None


def test_partial_or_gapped_authority_cannot_recover(tmp_path):
    tools=editor(tmp_path)
    result=apply(tools,[change()],[view(tools,1,1)])
    assert result['error']['code']=='no_match' and (tmp_path/'subject.py').read_text()==SOURCE
    source='def target(value):\n    interim = value - 1\n    return interim\n'
    (tmp_path/'subject.py').write_text(source)
    old='  def target( value ):\n    interim=value-1\n    return interim\n'
    new='  def target( value ):\n    interim=value+1\n    return interim\n'
    result=apply(tools,[change(old,new)],[view(tools,1,1),view(tools,3,3)])
    assert result['error']['code']=='no_match' and (tmp_path/'subject.py').read_text()==source


def test_adjacent_fresh_cards_cover_whole_function_without_broadening_authority(tmp_path):
    tools=editor(tmp_path)
    result=apply(tools,[change()],[view(tools,1,1),view(tools,2,2)])
    assert result['success'] and 'value+1' in (tmp_path/'subject.py').read_text()


def test_stale_hash_rejects_before_recovery(tmp_path):
    tools=editor(tmp_path);assert tools.register_presented([view(tools)])['accepted']==1
    (tmp_path/'subject.py').write_text(SOURCE.replace('value - 1','value - 2'))
    result=apply(tools,[change()])
    assert result['error']['code']=='stale_source'
    assert (tmp_path/'subject.py').read_text()==SOURCE.replace('value - 1','value - 2')


def test_recovered_scope_and_exact_inner_edit_overlap_atomically(tmp_path):
    tools=editor(tmp_path);before=(tmp_path/'subject.py').stat()
    result=apply(tools,[change(),change('return value - 1','return value - 3')],[view(tools)])
    assert result['error']['code']=='overlapping_edits'
    after=(tmp_path/'subject.py').stat()
    assert (tmp_path/'subject.py').read_text()==SOURCE and before.st_ino==after.st_ino and before.st_mtime_ns==after.st_mtime_ns


def test_recovery_forces_compile_for_entire_batch_before_any_write(tmp_path):
    tools=editor(tmp_path);before=(tmp_path/'subject.py').stat()
    result=apply(tools,[change(),{'path':'new/deep/bad.py','content':'def broken(:\n'}],[view(tools)])
    assert result['error']['code']=='python_syntax_error'
    assert (tmp_path/'subject.py').read_text()==SOURCE and not (tmp_path/'new').exists()
    after=(tmp_path/'subject.py').stat()
    assert before.st_ino==after.st_ino and before.st_mtime_ns==after.st_mtime_ns


def test_opt_out_keeps_original_exact_matching_behavior(tmp_path):
    tools=editor(tmp_path,enabled=False)
    result=apply(tools,[change()],[view(tools)])
    assert result['error']['code']=='no_match' and (tmp_path/'subject.py').read_text()==SOURCE
