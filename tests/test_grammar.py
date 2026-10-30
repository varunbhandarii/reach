import pytest
from reach.grammar import parse


@pytest.mark.parametrize('text,kind,phrase', [
    ('um please find the wallet please','find','wallet'), ("where's my wallet",'find','wallet'),
    ('find the pen please','find','pen'), ('wallet','find','wallet'), ('the apple','find','apple'),
    ('could you look for a ball','find','ball'), ('locate an apple','find','apple'),
    ('find wallett','find','wallet'), ('find blue wallett','find','blue wallet'),
    ('find my purple umbrella','find','purple umbrella'), ('stop','stop',None),
    ("don't stop",'unknown',None), ('do not find the wallet','unknown',None),
    ('the left one','choose','left'), ('right','choose','right'), ('yeah','yes',None),
    ('never mind','cancel',None), ('reset','start_over',None), ('say that again','repeat',None),
    ('repeat repeat','repeat',None), ('check again','recheck',None), ('which one','unknown',None),
    ('find wallet, pen','unknown',None), ('find '+('x'*121),'unknown',None), ('this','unknown',None),
])
def test_command_intents(text,kind,phrase):
    c=parse(text,t_mono=3,turn_id=4,source='key')
    assert (c.kind,c.phrase)==(kind,phrase)
    assert c.t_mono==3 and c.turn_id==4 and c.source=='key'
