"""Complete intermediate-only geometry gate for arbitrary frozen train cohorts."""
ACTIONS={'original_clean','original_roundtrip','target_pixels','target_roundtrip','opposite_pixels','opposite_roundtrip'}
def geometry_gate(report,expected_states):
    expected=set(expected_states);rows=report['rows'];keyed={}
    if not expected:raise ValueError('No localized train instances')
    for row in rows:
        key=(row['state_id'],row['action'])
        if key in keyed:raise ValueError('Duplicate geometry result')
        if not isinstance(row['correct'],(bool,int)) or row['correct'] not in (False,True,0,1):raise ValueError('Missing geometry correctness')
        keyed[key]=row
    if set(keyed)!={(sid,a) for sid in expected for a in ACTIONS}:raise ValueError('Incomplete/unexpected intermediate geometry')
    return {sid:bool(keyed[sid,'target_pixels']['correct'] and keyed[sid,'target_roundtrip']['correct']) for sid in expected}
