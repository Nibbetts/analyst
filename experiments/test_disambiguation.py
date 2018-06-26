# NOTE: This test didn't really succeed - though it could be useful for later
#   work with full-on clusters, and which ones we find ourselves near.
# NOTE: This is in a space with only the 20,000 most common or so words.

import analyst as an
import numpy as np


# Prepare stuff:

# NOTE: To be run from:
#   /mnt/pccfs/not_backed_up/Nathan/analyst_project_21_may_2018/
a = an.Analyst.load("saved_analyses/5_degrees_normalized_20000/an20000_glove_normalized.dill")

hubber = a.find_evaluator("Nodal 4-Hubs")
hubs = hubber.get_clusters()
hub_names = [h.name for h in hubs]

# Randomish selection of homonyms and homographs:
l = ['bank', 'pitch', 'pig', 'goal', 'program', 'mouse', 'keys', 'tear',
     'wind', 'seal', 'sink', 'park', 'match', 'invalid', 'crane', 'bow']

# NOTE: we're not using the built-in downstream's k'th neighbor starter simply
#   because k'th neighbors are not yet implemented as I'm writing this test,
#   and we don't need the input word appended to the beginning.
downstream_pair = lambda w: a.downstream(w, start_neighbor_k=0, give_path=False)

def disambiguation(word, n=30):
    v=a.as_vector(word)
    dists=np.dot(a.space,v)
    inds=np.argsort(dists)[::-1]
    words=list(map(a.as_string,inds))
    downs=list(map(downstream_pair,words))
    downs=list(map(np.sort,downs)) # Sort else sometimes flipped.
    downs=list(map(tuple,downs))
    for i, t in enumerate(downs[:n]):
        hub_name = ""
        if t[0] in hub_names:
            hub_name = t[0]
        if t[1] in hub_names:
            if hub_name != "": hub_name += ", "
            hub_name += t[1]
        print("{:<14}".format(words[i]), "{:<30}".format(str(t)), hub_name)
        # Output columns:
        # n'th neighbor, its downstream Node, names of hubs that Node is in.
    
# Now show stuff:

print(len(hubs))

hub_nodes = [h.nodes[0] for h in hubs]
print(len(hub_nodes))

strings = [str(n) for n in hub_nodes]
print(len(set(strings)))
print("")

for w in l:
    print(w + ":\n")
    disambiguation(w)
    print("\n")
 

# OUTPUT:
"""
1432
1432
1283

bank:

bank           ('bank', 'banks')              
banks          ('bank', 'banks')              
banking        ('bank', 'banks')              
central        ('eastern', 'western')         
credit         ('loan', 'loans')              
bankers        ('executives', 'managers')     executives
financial      ('economic', 'economy')        economic
investment     ('investment', 'investments')  investment
lending        ('borrowing', 'lending')       
citibank       ('citigroup', 'jpmorgan')      citigroup
monetary       ('imf', 'monetary')            
loans          ('loan', 'loans')              
lender         ('borrowers', 'lenders')       
securities     ('brokerage', 'brokerages')    brokerage
funds          ('fund', 'funds')              
finance        ('minister', 'prime')          
hsbc           ('abn', 'amro')                
deposit        ('deposit', 'deposits')        
west           ('east', 'west')               
institutions   ('institution', 'institutions') 
accounts       ('account', 'accounts')        
cash           ('fund', 'funds')              
money          ('fund', 'funds')              
currency       ('currencies', 'currency')     currencies, currency
citigroup      ('citigroup', 'jpmorgan')      citigroup
palestinian    ('israelis', 'palestinians')   
fund           ('fund', 'funds')              
savings        ('pension', 'pensions')        
fed            ('bernanke', 'greenspan')      
deposits       ('deposit', 'deposits')        


pitch:

pitch          ('pitch', 'pitches')           
pitches        ('pitch', 'pitches')           
inning         ('homer', 'inning')            inning
fastball       ('pitch', 'pitches')           
pitching       ('pitcher', 'pitchers')        
ball           ('ball', 'balls')              
pitcher        ('pitcher', 'pitchers')        
pitched        ('homer', 'inning')            inning
mound          ('mound', 'mounds')            
infield        ('infield', 'outfield')        
hitter         ('hitter', 'hitters')          hitter
bat            ('bat', 'bats')                
innings        ('homer', 'inning')            inning
hitters        ('hitter', 'hitters')          hitter
outs           ('homer', 'inning')            inning
pitchers       ('pitcher', 'pitchers')        
outfield       ('infield', 'outfield')        
game           ('game', 'games')              
bullpen        ('pitcher', 'pitchers')        
throw          ('threw', 'thrown')            threw
field          ('field', 'fields')            
sox            ('mets', 'yankees')            mets, yankees
batters        ('homer', 'inning')            inning
yankees        ('mets', 'yankees')            mets, yankees
pinch          ('hitter', 'hitters')          hitter
bounce         ('bounced', 'bouncing')        bounced
hander         ('pitcher', 'pitchers')        
hit            ('hit', 'hitting')             hit
starter        ('starter', 'starters')        
got            ('get', 'getting')             get


pig:

pig            ('chickens', 'poultry')        
pigs           ('chickens', 'poultry')        
hog            ('chickens', 'poultry')        
cow            ('goats', 'sheep')             
sheep          ('goats', 'sheep')             
chicken        ('beef', 'meat')               meat
chickens       ('chickens', 'poultry')        
goat           ('goats', 'sheep')             
animal         ('animal', 'animals')          animals
pork           ('beef', 'meat')               meat
meat           ('beef', 'meat')               meat
cattle         ('cattle', 'livestock')        
cows           ('goats', 'sheep')             
rabbit         ('bug', 'bugs')                
goats          ('goats', 'sheep')             
animals        ('animal', 'animals')          animals
dog            ('dog', 'dogs')                dog
poultry        ('chickens', 'poultry')        
duck           ('duck', 'lame')               
monkey         ('monkey', 'monkeys')          
cat            ('dog', 'dogs')                dog
rat            ('mice', 'rats')               mice
farm           ('farm', 'farms')              
pet            ('pet', 'pets')                
goose          ('duck', 'lame')               
dogs           ('dog', 'dogs')                dog
cats           ('dog', 'dogs')                dog
mouth          ('mouth', 'mouths')            mouth
bird           ('bird', 'birds')              birds
herd           ('goats', 'sheep')             


goal:

goal           ('goal', 'goals')              
goals          ('goal', 'goals')              
scored         ('scored', 'scoring')          scored
scoring        ('scored', 'scoring')          scored
kick           ('kick', 'kicks')              
minute         ('minute', 'minutes')          
penalty        ('penalties', 'penalty')       
half           ('almost', 'nearly')           
header         ('crossbar', 'header')         
striker        ('midfielder', 'striker')      midfielder, striker
score          ('scored', 'scoring')          scored
chances        ('chance', 'chances')          
equalizer      ('equalized', 'equalizer')     
substitute     ('substitute', 'substituted')  
chance         ('chance', 'chances')          
scorer         ('scorer', 'scorers')          
missed         ('missed', 'misses')           
victory        ('victory', 'win')             victory
forward        ('forward', 'forwards')        
2-0            ('2-0', '3-0')                 2-0
lead           ('lead', 'leads')              
minutes        ('minute', 'minutes')          
assist         ('assist', 'assisting')        
effort         ('effort', 'efforts')          
aim            ('aim', 'aims')                aim
game           ('game', 'games')              
second         ('second', 'third')            second
put            ('put', 'putting')             
win            ('victory', 'win')             victory
tying          ('tie', 'tied')                


program:

program        ('program', 'programs')        
programs       ('program', 'programs')        
programme      ('programme', 'programmes')    
programmes     ('programme', 'programmes')    
education      ('education', 'educational')   
project        ('project', 'projects')        projects
plan           ('plan', 'plans')              
educational    ('education', 'educational')   
addition       ('include', 'including')       include
administration ('government', 'governments')  government
funding        ('financing', 'funding')       
development    ('project', 'projects')        projects
provide        ('provide', 'providing')       provide
assistance     ('aid', 'assistance')          aid
projects       ('project', 'projects')        projects
provides       ('provide', 'providing')       provide
training       ('trained', 'training')        
initiative     ('initiative', 'initiatives')  
effort         ('effort', 'efforts')          
outreach       ('non-profit', 'nonprofit')    nonprofit
plans          ('plan', 'plans')              
work           ('work', 'working')            
programming    ('broadcast', 'broadcasts')    broadcast
system         ('system', 'systems')          
part           ('although', 'though')         although, though
curriculum     ('course', 'courses')          courses
study          ('studies', 'study')           
students       ('student', 'students')        students
show           ('show', 'shows')              
providing      ('provide', 'providing')       provide


mouse:

mouse          ('mice', 'rats')               mice
mice           ('mice', 'rats')               mice
rat            ('mice', 'rats')               mice
keyboard       ('keyboard', 'keyboards')      
rabbit         ('bug', 'bugs')                
monkey         ('monkey', 'monkeys')          
rats           ('mice', 'rats')               mice
cat            ('dog', 'dogs')                dog
monkeys        ('monkey', 'monkeys')          
click          ('button', 'buttons')          
cartoon        ('cartoon', 'cartoons')        
buttons        ('button', 'buttons')          
bugs           ('bug', 'bugs')                
frog           ('frog', 'frogs')              
bunny          ('bug', 'bugs')                
worm           ('worm', 'worms')              
mickey         ('mantle', 'mickey')           
animated       ('cartoon', 'cartoons')        
screen         ('screen', 'screens')          
spider         ('insect', 'insects')          insects
button         ('button', 'buttons')          
user           ('user', 'users')              
pig            ('chickens', 'poultry')        
miniature      ('miniature', 'replica')       
robot          ('robot', 'robots')            
computer       ('computer', 'computers')      computer
interface      ('functionality', 'interface') interface
worms          ('worm', 'worms')              
keyboards      ('keyboard', 'keyboards')      
ears           ('ear', 'ears')                


keys:

keys           ('keyboard', 'keyboards')      
keyboard       ('keyboard', 'keyboards')      
buttons        ('button', 'buttons')          
key            ('crucial', 'vital')           crucial, vital
alicia         ('amy', 'jennifer')            amy, jennifer
fingers        ('finger', 'fingers')          fingers
strings        ('string', 'strings')          
keyboards      ('keyboard', 'keyboards')      
button         ('button', 'buttons')          
door           ('door', 'doors')              
wallet         ('bag', 'bags')                bag
lock           ('lock', 'locks')              lock
piano          ('cello', 'violin')            cello
locking        ('lock', 'locks')              lock
hand           ('hand', 'hands')              
doors          ('door', 'doors')              
guitar         ('guitar', 'guitars')          guitar
locks          ('lock', 'locks')              lock
horns          ('horn', 'horns')              
pocket         ('pocket', 'pockets')          
functions      ('function', 'functions')      function
mouse          ('mice', 'rats')               mice
phones         ('phones', 'telephones')       phones
switch         ('switch', 'switching')        
boxes          ('box', 'boxes')               
ring           ('ring', 'rings')              
your           ('my', 'your')                 
switches       ('switch', 'switching')        
florida        ('florida', 'miami')           
vital          ('crucial', 'vital')           crucial, vital


tear:

tear           ('disperse', 'dispersed')      
disperse       ('disperse', 'dispersed')      
riot           ('demonstrators', 'protesters') demonstrators
bullets        ('bullet', 'bullets')          bullet
demonstrators  ('demonstrators', 'protesters') demonstrators
protesters     ('demonstrators', 'protesters') demonstrators
grenades       ('grenade', 'grenades')        grenades
cannons        ('cannon', 'cannons')          
rioters        ('mobs', 'rioters')            
tearing        ('ripped', 'tore')             tore
throwing       ('threw', 'thrown')            threw
protestors     ('demonstrators', 'protesters') demonstrators
hurled         ('threw', 'thrown')            threw
sprayed        ('sprayed', 'spraying')        
fired          ('fired', 'firing')            
spray          ('sprayed', 'spraying')        
dispersed      ('disperse', 'dispersed')      
firing         ('fired', 'firing')            
stones         ('stone', 'stones')            
grenade        ('grenade', 'grenades')        grenades
gas            ('crude', 'oil')               crude, oil
marchers       ('demonstrators', 'protesters') demonstrators
tore           ('ripped', 'tore')             tore
crowds         ('crowd', 'crowds')            crowd
bullet         ('bullet', 'bullets')          bullet
youths         ('teenagers', 'teens')         teenagers
cannon         ('cannon', 'cannons')          
shields        ('helmet', 'helmets')          
rubber         ('bag', 'bags')                bag
burning        ('burned', 'burnt')            


wind:

wind           ('wind', 'winds')              winds
winds          ('wind', 'winds')              winds
blowing        ('blew', 'blowing')            
storms         ('hurricane', 'storm')         hurricane, storm
storm          ('hurricane', 'storm')         hurricane, storm
turbines       ('turbine', 'turbines')        turbine
rain           ('rains', 'torrential')        rains
weather        ('rains', 'torrential')        rains
breeze         ('wind', 'winds')              winds
waves          ('wave', 'waves')              waves
solar          ('gasoline', 'petrol')         gasoline
currents       ('wave', 'waves')              waves
temperatures   ('temperature', 'temperatures') temperature
clouds         ('cloud', 'clouds')            clouds
mph            ('kph', 'mph')                 
gale           ('wind', 'winds')              winds
humidity       ('temperature', 'temperatures') temperature
turbine        ('turbine', 'turbines')        turbine
swirling       ('cloud', 'clouds')            clouds
speeds         ('speed', 'speeds')            
water          ('drink', 'drinks')            drink, drinks
snow           ('rains', 'torrential')        rains
heat           ('temperature', 'temperatures') temperature
electricity    ('utilities', 'utility')       
hurricane      ('hurricane', 'storm')         hurricane, storm
trees          ('tree', 'trees')              
sails          ('sail', 'sailing')            sail
rains          ('rains', 'torrential')        rains
blows          ('blow', 'blows')              
dust           ('cloud', 'clouds')            clouds


seal:

seal           ('seal', 'seals')              
seals          ('seal', 'seals')              
sealing        ('sealed', 'sealing')          
sealed         ('sealed', 'sealing')          
wrap           ('wrapped', 'wrapping')        
hunt           ('hunters', 'hunting')         hunters, hunting
stamped        ('envelope', 'stamped')        
whale          ('whale', 'whales')            whale
stamp          ('postage', 'stamps')          
coat           ('coat', 'coats')              
dolphin        ('whale', 'whales')            whale
meat           ('beef', 'meat')               meat
secured        ('secure', 'securing')         
secure         ('secure', 'securing')         
fur            ('coat', 'coats')              
clinch         ('clinch', 'clinched')         
elephant       ('elephant', 'elephants')      
whales         ('whale', 'whales')            whale
order          ('order', 'orders')            
sheet          ('sheet', 'sheets')            
preserve       ('preserve', 'preserving')     
prevent        ('prevent', 'preventing')      preventing
golden         ('award', 'awards')            
plastic        ('bag', 'bags')                bag
remove         ('remove', 'removing')         removing
protect        ('protect', 'protecting')      protect
bear           ('bear', 'bears')              
protected      ('protect', 'protecting')      protect
bears          ('bear', 'bears')              
sea            ('ocean', 'sea')               ocean


sink:

sink           ('sink', 'sinks')              
sinks          ('sink', 'sinks')              
sinking        ('sank', 'sinking')            
sunk           ('sank', 'sinking')            
sank           ('sank', 'sinking')            
bottom         ('also', 'both')               both
drain          ('drainage', 'irrigation')     
bathroom       ('toilet', 'toilets')          
kitchen        ('toilet', 'toilets')          
plunge         ('plummeted', 'plunged')       
water          ('drink', 'drinks')            drink, drinks
boat           ('boat', 'boats')              boat
shallow        ('deep', 'deeper')             deep
deck           ('roof', 'roofs')              roof
toilet         ('toilet', 'toilets')          
ship           ('ship', 'vessel')             ship
float          ('float', 'floating')          
explode        ('explode', 'exploding')       
boats          ('boat', 'boats')              boat
dry            ('dry', 'wet')                 dry
vessel         ('ship', 'vessel')             ship
deeper         ('deep', 'deeper')             deep
disappear      ('faded', 'fading')            
drains         ('drainage', 'irrigation')     
ocean          ('ocean', 'sea')               ocean
torpedo        ('submarine', 'submarines')    
melt           ('melt', 'melted')             
surface        ('surface', 'surfaces')        
shower         ('toilet', 'toilets')          
floating       ('float', 'floating')          


park:

park           ('park', 'parks')              
parks          ('park', 'parks')              
recreation     ('recreation', 'recreational') 
forest         ('forest', 'forests')          
amusement      ('attraction', 'attractions')  
gardens        ('garden', 'gardens')          gardens
adjacent       ('adjacent', 'adjoining')      
avenue         ('avenue', 'boulevard')        avenue
located        ('located', 'situated')        
garden         ('garden', 'gardens')          gardens
area           ('area', 'areas')              
stadium        ('arena', 'stadium')           stadium
south          ('north', 'south')             
grove          ('oak', 'pine')                oak, pine
riverside      ('bernardino', 'riverside')    
road           ('highways', 'roads')          
hill           ('beverly', 'hills')           
site           ('site', 'sites')              
wildlife       ('conservation', 'wildlife')   conservation
nearby         ('near', 'nearby')             
north          ('north', 'south')             
memorial       ('monument', 'monuments')      monument, monuments
lake           ('lake', 'lakes')              
museum         ('galleries', 'gallery')       
visitors       ('tourists', 'visitors')       
near           ('near', 'nearby')             
yellowstone    ('deer', 'elk')                elk
arlington      ('va.', 'virginia')            va.
attraction     ('attraction', 'attractions')  
campus         ('campus', 'campuses')         


match:

match          ('match', 'matches')           
matches        ('match', 'matches')           
tournament     ('tournament', 'tournaments')  
final          ('second', 'third')            second
semifinal      ('quarterfinal', 'semifinal')  
quarterfinal   ('quarterfinal', 'semifinal')  
game           ('game', 'games')              
play           ('played', 'playing')          
champions      ('quarterfinals', 'semifinals') 
draw           ('draw', 'draws')              
finals         ('quarterfinals', 'semifinals') 
round          ('round', 'rounds')            
championship   ('championship', 'championships') 
cup            ('cup', 'cups')                
win            ('victory', 'win')             victory
qualifier      ('qualifier', 'qualifiers')    
qualifying     ('qualifier', 'qualifiers')    
tie            ('tie', 'tied')                
playing        ('played', 'playing')          
players        ('player', 'players')          
cricket        ('cricket', 'rugby')           cricket
2-1            ('1-0', '2-1')                 
played         ('played', 'playing')          
against        ('opponent', 'opponents')      
scoring        ('scored', 'scoring')          scored
2-0            ('2-0', '3-0')                 2-0
1-0            ('1-0', '2-1')                 
clash          ('clashes', 'confrontations')  clashes
quarterfinals  ('quarterfinals', 'semifinals') 
opening        ('opened', 'opening')          


invalid:

invalid        ('invalid', 'valid')           
valid          ('invalid', 'valid')           
unconstitutional ('amendment', 'amendments')    amendment
ineligible     ('eligible', 'ineligible')     
void           ('fill', 'filling')            fill
legally        ('allow', 'allowing')          
fraudulent     ('fraud', 'fraudulent')        fraud
illegitimate   ('lawful', 'unlawful')         
ballots        ('ballot', 'ballots')          ballots
meaningless    ('irrelevant', 'meaningless')  
irrelevant     ('irrelevant', 'meaningless')  
useless        ('irrelevant', 'meaningless')  
incorrect      ('inaccurate', 'misleading')   inaccurate
ballot         ('ballot', 'ballots')          ballots
declared       ('declare', 'declaring')       declaring
validity       ('legality', 'validity')       
rendered       ('rendered', 'rendering')      
insufficient   ('inadequate', 'insufficient') 
unfit          ('irresponsible', 'reckless')  irresponsible
deemed         ('considered', 'regarded')     regarded
flawed         ('dramatically', 'drastically') 
signatures     ('petition', 'petitions')      petition
disqualified   ('prosecute', 'prosecuting')   
declaring      ('declare', 'declaring')       declaring
unfounded      ('baseless', 'unfounded')      
unlawful       ('lawful', 'unlawful')         
unreliable     ('exact', 'precise')           exact, precise
obsolete       ('obsolete', 'outdated')       
false          ('inaccurate', 'misleading')   inaccurate
technically    ('allow', 'allowing')          


crane:

crane          ('crane', 'cranes')            
cranes         ('crane', 'cranes')            
barge          ('boat', 'boats')              boat
hook           ('hook', 'hooks')              
elevator       ('staircase', 'stairs')        
tow            ('boat', 'boats')              boat
lift           ('lift', 'lifting')            lift
ladder         ('rope', 'ropes')              
hooks          ('hook', 'hooks')              
dock           ('dock', 'docks')              
piers          ('pier', 'wharf')              pier
tower          ('tower', 'towers')            
steel          ('stainless', 'steel')         steel
hydraulic      ('electrical', 'mechanical')   
eagle          ('bald', 'eagle')              eagle
tractor        ('tractor', 'tractors')        
rope           ('rope', 'ropes')              
helicopter     ('helicopter', 'helicopters')  helicopter
floating       ('float', 'floating')          
bird           ('bird', 'birds')              birds
collapses      ('collapsed', 'collapsing')    collapsing
operator       ('operator', 'operators')      
roof           ('roof', 'roofs')              roof
truck          ('cars', 'vehicles')           cars, vehicles
boat           ('boat', 'boats')              boat
bell           ('bell', 'bells')              
birds          ('bird', 'birds')              birds
lifted         ('lift', 'lifting')            lift
ship           ('ship', 'vessel')             ship
griffin        ('mitchell', 'ross')           


bow:

bow            ('arrow', 'arrows')            
arrow          ('arrow', 'arrows')            
bend           ('bending', 'twisting')        bending
bowed          ('cried', 'wept')              wept
stern          ('george', 'w.')               
arrows         ('arrow', 'arrows')            
hull           ('middlesbrough', 'sunderland') sunderland
boat           ('boat', 'boats')              boat
wow            ('ok', 'okay')                 
wear           ('wearing', 'wore')            wearing, wore
neck           ('chest', 'neck')              chest
door           ('door', 'doors')              
fitted         ('equipped', 'fitted')         
sword          ('sword', 'swords')            
skirt          ('skirt', 'skirts')            
tail           ('tail', 'tails')              
pin            ('pin', 'pins')                
jiabao         ('jiabao', 'wen')              
trousers       ('pants', 'trousers')          pants, trousers
wooden         ('wood', 'wooden')             wooden
wearing        ('wearing', 'wore')            wearing, wore
waist          ('pants', 'trousers')          pants, trousers
tie            ('tie', 'tied')                
hand           ('hand', 'hands')              
skirts         ('skirt', 'skirts')            
torpedo        ('submarine', 'submarines')    
jacket         ('jacket', 'jackets')          
dress          ('skirt', 'skirts')            
hair           ('blond', 'blonde')            blond
deck           ('roof', 'roofs')              roof

"""