# isic
BACKGROUND_CATEGORY = ['Normal skin','Black pores','Hair','Scale line','regular but incomplete circle',
                       'Orange, blue, green, and bright yellow circular and semicircular areas.','ruler']

class_names = ['Pathological skin','abnormal skin']
                   
new_class_names = ['Pathological skin','abnormal skin']

# lung
BACKGROUND_CATEGORY_lung = ['non-X-ray irradiated area','X-ray exposure area of the body outside the thoracic cavity']

class_names_lung = ['left lung', 'right lung']

new_class_names_lung = ['left lung', 'right lung']

# deepglobe   Urban land ,Agriculture land,Rangeland,Forest land,Water,Barren land,Unknown
# Agriculture land: Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations.
# Urban land: building structure
# Rangeland: Any non-forest, non-farm, green land, grass.
# Forest land: Any land with at least 20% tree crown density plus clear cuts.
# Water: Rivers, oceans, lakes, wetland, ponds.
# Barren land: Mountain, rock, dessert, beach, land with no vegetation.
# Unknown: Clouds and others.
BACKGROUND_CATEGORY_D1 = [
    'Urban land','building','structure',
    'Range land','green land','grass',
    'Forest land','Any land with at least 20% tree crown density plus clear cuts'
    'Water','Rivers','oceans','lakes','wetland','ponds',
    'Barren land','Mountain','rock','dessert','beach','land with no vegetation',
    'Clouds',

]

class_names_D1 = ["Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations",'Agriculture land', 'Farms',
                  'any planned plantation','cropland','orchards','vineyards','nurseries','ornamental horticultural areas','confined feeding operations']

new_class_names_D1 = ["Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations",'Agriculture land',
                      'Farms','any planned plantation','cropland','orchards','vineyards','nurseries','ornamental horticultural areas','confined feeding operations']

BACKGROUND_CATEGORY_D2 = [
    "Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations",'Agriculture land',
    'Farms','any planned plantation','cropland','orchards','vineyards','nurseries','ornamental horticultural areas','confined feeding operations'
    'Range land','Any non-forest','non-farm','green land','grass',
    'Forest land','Any land with at least 20% tree crown density plus clear cuts'
    'Water','Rivers','oceans','lakes','wetland','ponds',
    'Barren land','Mountain','rock','dessert','beach','land with no vegetation',
    'Clouds',
]

class_names_D2 = ['Urban land','building','structure']

new_class_names_D2 = ['Urban land','building','structure']


BACKGROUND_CATEGORY_D3 = [
    "Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations",
    'Agriculture land', 'Farms','any planned plantation','cropland','orchards','vineyards','nurseries','ornamental horticultural areas','confined feeding operations'
    'Urban land','building','structure',
    'Forest land','Any land with at least 20% tree crown density plus clear cuts'
    'Water','Rivers','oceans','lakes','wetland','ponds',
    'Barren land','Mountain','rock','dessert','beach','land with no vegetation',
    'Clouds',
]

class_names_D3 = ['Range land','green land','grass']

new_class_names_D3 = ['Range land','green land','grass']

BACKGROUND_CATEGORY_D4 = [
    "Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations",
    'Agriculture land', 'Farms','any planned plantation','cropland','orchards','vineyards','nurseries','ornamental horticultural areas','confined feeding operations'
    'Urban land','building','structure',
    'Range land','green land','grass'
    'Water','Rivers','oceans','lakes','wetland','ponds',
    'Barren land','Mountain','rock','dessert','beach','land with no vegetation',
    'Clouds',
]

class_names_D4 = ['Forest land','Forest']

new_class_names_D4 = ['Forest land','Forest']

BACKGROUND_CATEGORY_D5 = [
    "Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations",'Agriculture land', 'Farms',
    'any planned plantation','cropland','orchards','vineyards','nurseries','ornamental horticultural areas','confined feeding operations'
    'Urban land','building','structure',
    'Range land','green land','grass',
    'Forest land','Forest',
    'Barren land','Mountain','rock','dessert','beach','land with no vegetation',
    'Clouds',
]

class_names_D5 = ['Water','Rivers','oceans','lakes','wetland','ponds']

new_class_names_D5 = ['Water','Rivers','oceans','lakes','wetland','ponds']

BACKGROUND_CATEGORY_D6 = [
    "Agriculture land:Farms, any planned plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas;"
    " confined feeding operations",'Agriculture land', 'Farms','any planned plantation','cropland','orchards','vineyards','nurseries',
    'ornamental horticultural areas','confined feeding operations'
    'Urban land','building','structure',
    'Range land','green land','grass',
    'Forest land','Forest',
    'Water','Rivers','oceans','lakes','wetland','ponds',
    'Clouds',
]

class_names_D6 = ['Barren land','Mountain','rock','dessert','beach','land with no vegetation']

new_class_names_D6 = ['Barren land','Mountain','rock','dessert','beach','land with no vegetation']



class_names_fss = ['abacus', "abe's flyingfish", 'ab wheel', 'accordion', 'acorn', 'ac ground', 'ac wall', 'adhensive tape',
                        'adidas logo1', 'adidas logo2', 'afghan hound', 'african crocodile', 'african elephant', 'african grey',
                        'agama', 'aircraft carrier', 'airedale', 'airliner', 'airship', 'air strip', 'albatross', 'almond', 'ambulance',
                        'american alligator', 'american chamelon', 'american staffordshire', 'andean condor', 'anemone fish', 'angora',
                        'anise', 'apple', 'apple icon', 'apron', 'arabian camel', 'arch bridge', 'arctic fox', 'armadillo', 'armour',
                        'arrow', 'artichoke', 'ashtray', 'assult rifle', 'astronaut', 'aubergine', 'australian terrier', 'avocado', 'baboon',
                        'baby', 'backpack', 'badger', 'bagel', 'balance beam', 'balance weight', 'bald eagle', 'balloon', 'ballpoint', 'bamboo dragonfly',
                        'bamboo slip', 'banana', 'banana boat', 'band-aid', 'banded gecko', 'banjo', 'barbell', 'barber shaver',
                        'barometer', 'baseball', 'baseball bat', 'baseball player', 'basketball', 'basset', 'bassoon', 'bat',
                        'bathtub', 'bath ball', 'battery', 'beacon', 'beagle', 'beaker', 'beam bridge', 'bear', 'beaver', 'bedlington terrier',
                        'bee', 'beer bottle', 'beer glass', 'beet root', 'bee eater', 'bee house', 'bell', 'bell pepper', 'besom', 'bighorn sheep',
                        'big ben', 'birdhouse', 'bison', 'bittern', 'black bear', 'black grouse', 'black stork', 'black swan', 'blenheim spaniel',
                        'bloodhound', 'blossom card', 'bluetick', 'boa constrictor', 'bolete', 'bolotie', 'bomb', 'border terrier', 'boston bull',
                        'bottle cap', 'bouzouki', 'bowtie', 'boxing gloves', 'box turtle', 'bra', 'bracelet', 'bradypod', 'brain coral', 'brambling',
                        'brasscica', 'breast pump', 'briard', 'brick', 'brick card', 'brick tea', 'briefcase', 'broccoli', 'broom', 'brown bear',
                        'brush pen', 'bucket', 'buckingham palace', 'buckler', 'bulb', 'bulbul bird', 'bullet train', 'burj al', 'bus', 'bushtit',
                        'bustard', 'butterfly', 'cabbage', 'cabbage butterfly', 'cablestayed bridge', 'cableways', 'cactus', 'cactus ball', 'cairn',
                        'calculator', 'camel', 'camomile', 'candle', 'cannon', 'canoe', 'cantilever bridge', 'canton tower', 'can opener', 'captain america shield',
                        'capuchin', 'carambola', 'carbonara', 'cardoon', 'carousel', 'carp', 'carriage', 'carrot', 'carton', 'car mirror', 'car wheel', 'cassette',
                        'cathedrale paris', 'cauliflower', 'cd', 'ceiling fan', 'celery', 'cello', 'chainsaw', 'chalk', 'chalk brush', 'chandelier', 'charge battery',
                        'cheese', 'cheese burger', 'cheetah', 'cherry', 'chess bishop', 'chess king', 'chess knight', 'chess queen', 'chest', 'chickadee bird',
                        'chicken', 'chicken leg', 'chicken wings', 'chicory', 'chiffon cake', 'chihuahua', 'children slide', 'chimpanzee', 'chinese date',
                        'chinese knot', 'chopsticks', 'christmas stocking', 'church', 'cicada', 'cigar', 'cigarette', 'clam', 'clearwing flyingfish', 'cleaver',
                        'cloud', 'cn tower', 'cocacola', 'cocktail shaker', 'coconut', 'coffeepot', 'coffee mug', 'coffin', 'coho', 'coin', 'collar', 'colubus',
                        'comb', 'combination lock', 'common newt', 'computer mouse', 'conch', 'condor', 'consomme', 'conversion plug', 'convertible', 'conveyor',
                        'corn', 'cornet', 'cornmeal', 'cosmetic brush', 'cottontail', 'coucal', 'cougar', 'cowboy hat', 'coyote', 'cpu', 'crab', 'cradle', 'crane',
                        'crash helmet', 'crayon', 'cream', 'crepe', 'cricket', 'cricketball', 'cristo redentor', 'crocodile', 'croissant', 'croquet ball',
                        'crt screen', 'cuckoo', 'cucumber', 'cumquat', 'cup', 'curlew', 'cushion', 'daisy', 'dandie dinmont', 'dart', 'dart target', 'delta wing',
                        'dhole', 'diamond', 'diaper', 'digital clock', 'digital watch', 'dingo', 'dinosaur', 'dishwasher', 'diver', 'donkey', 'doormat', 'doublebus',
                        'dough', 'doughnut', 'dowitcher', 'downy pitch', 'dragonfly', 'drake', 'drilling platform', 'drum', 'drumstick', 'dugong', 'dumbbell', 'dutch oven',
                        'dwarf beans', 'eagle', 'earphone1', 'earphone2', 'earplug', 'echidna', 'eel', 'eft newt', 'egg', 'eggnog', 'egg tart', 'egret', 'egyptian cat',
                        'electric fan', 'electronic stove', 'electronic toothbrush', 'eletrical switch', 'english foxhound', 'english setter', 'envelope', 'equestrian helmet',
                        'esport chair', 'espresso', 'espresso maker', 'excavator', 'f1 racing', 'face powder', 'fan', 'feather clothes', 'feeder', 'fennel bulb', 'ferrari911',
                        'ferret', 'fig', 'file cabinet', 'fire balloon', 'fire engine', 'fire hydrant', 'fire screen', 'fish', 'fish eagle', 'flamingo', 'flat-coated retriever',
                        'flatworm', 'flowerpot', 'flute', 'fly', 'flying disc', 'flying frog', 'flying geckos', 'flying snakes', 'flying squirrel', 'folding chair', 'fork', 'forklift',
                        'fountain', 'fox', 'fox squirrel', 'french ball', 'french fries', 'frog', 'frying pan', 'fur coat', 'ganeva chair', 'garbage can', 'garbage truck', 'garfish',
                        'garlic', 'gas pump', 'gas tank', 'gazelle', 'gecko', 'german pointer', 'giant panda', 'giant schnauzer', 'gibbon', 'ginger', 'glider flyingfish', 'gliding lizard',
                        'globe', 'goblet', 'golden plover', 'golden retriever', 'goldfinch', 'goldfish', 'golfcart', 'golf ball', 'goose', 'gorilla', 'gourd', 'grasshopper', 'great wall',
                        'green mamba', 'grey fox', 'grey whale', 'groenendael', 'guacamole', 'guinea pig', 'guitar', 'gym ball', 'gypsy moth', 'gyromitra', 'haddock', 'hair drier',
                        'hair razor', 'hami melon', 'hammer', 'hammerhead shark', 'hamster', 'handcuff', 'handkerchief', 'handshower', 'hang glider', 'hard disk', 'hare', 'harmonica',
                        'harp', 'hartebeest', 'harvester', 'har gow', 'hatchet', 'hawk', 'hawthorn', 'head cabbage', 'helicopter', 'hen of the woods', 'hippo', 'hock', 'hook', 'hornbill',
                        'hornet', 'hotdog', 'hotel slipper', 'housefinch', 'hover board', 'howler monkey', 'hummingbird', 'hyena', 'ibex', 'iceberg', 'icecream', 'ice lolly', 'igloo',
                        'iguana', 'impala', 'indian cobra', 'indian elephant', 'indri', 'ipad', 'iphone', 'ipod', 'ironing board', 'iron man', 'jacamar', 'jackfruit', 'jacko lantern',
                        'jay bird', 'jellyfish', 'jet aircraft', 'jinrikisha', 'jordan logo', 'joystick', 'kangaroo', 'kappa logo', 'kart', 'kazoo', 'key', 'keyboard', 'killer whale',
                        'kinguin', 'kitchen knife', 'kite', 'kit fox', 'knife', 'koala', 'kobe logo', 'kremlin', 'kunai', 'kwanyin', 'lacewing', 'ladder', 'ladle', 'ladybug', 'ladyfinger',
                        'lady slipper', 'lampshade', 'langur', 'laptop', 'lapwing', 'lark', 'lawn mower', 'leafhopper', 'leaf egg', 'leaf fan', 'leatherback turtle', 'leather shoes',
                        'leeks', 'leggings', 'lemon', 'lemur catta', 'leopard', 'lesser panda', 'letter opener', 'lettuce', 'lhasa apso', 'lifeboat', 'light tube', 'lion', 'lionfish',
                        'lipstick', 'litchi', 'little blue heron', 'llama', 'loafer', 'lobster', 'loggerhead turtle', 'loguat', 'lorikeet', 'lotus', 'louvre pyramid', 'lycaenid butterfly',
                        'lynx', 'macaque', 'macaw', 'magpie bird', 'mailbox', 'manatee', 'mango', 'manx', 'maotai bottle', 'maraca', 'marimba', 'mario', 'marmot', 'marshmallow',
                        'mashed potato', 'matchstick', 'may bug', 'mcdonald sign', 'mcdonald uncle', 'measuring cup', 'meatloaf', 'medical kit', 'meerkat', 'melon seed', 'memory stick',
                        'microphone', 'microscope', 'microsd', 'microwave', 'military vest', 'minicooper', 'miniskirt', 'mink', 'missile', 'mite predator', 'mitten', 'modem', 'moist proof pad',
                        'monarch butterfly', 'mongoose', 'monitor', 'monkey', 'monocycle', 'mooli', 'moon', 'mooncake', 'mortar', 'motarboard', 'motorbike', 'motor scooter', 'mountain tent',
                        'mount fuji', 'mouse', 'mouthpiece', 'mud turtle', 'mule', 'muscle car', 'mushroom', 'nagoya castle', 'nail scissor', 'narcissus', 'necklace', 'neck brace', 'nematode',
                        'net surface shoes', 'night snake', 'nike logo', 'nintendo 3ds', 'nintendo gba', 'nintendo sp', 'nintendo switch', 'nintendo wiiu', 'obelisk', 'ocarina', 'ocicat',
                        'oiltank car', 'oil filter', 'okra', 'olive', 'one-armed bandit', 'onion', 'orang', 'orange', 'oriole', 'oscilloscope', 'osprey', 'ostrich', 'otter', 'owl', 'ox',
                        'oyster', 'paddle', 'paint brush', 'panda', 'panpipe', 'panther', 'papaya', 'paper crane', 'paper plane', 'paper towel', 'parachute', 'parallel bars', 'parking meter',
                        'park bench', 'parthenon', 'partridge', 'patas', 'pay phone', 'peacock', 'peanut', 'pear', 'pen', 'pencil box', 'pencil sharpener1', 'pencil sharpener2', 'pepitas',
                        'peregine falcon', 'perfume', 'persian cat', 'persimmon', 'petri dish', 'pheasant', 'phonograph', 'photocopier', 'piano keyboard', 'pickelhaube', 'pickup', 'pidan',
                        'pig', 'pillow', 'pill bottle', 'pineapple', 'pinecone', 'pingpong ball', 'pingpong racket', 'pinwheel', 'pistachio', 'pizza', 'plaice', 'plastic bag', 'plate',
                        'platypus', 'poached egg', 'poker', 'pokermon ball', 'polar bear', 'polecat', 'police car', 'polo shirt', 'pomegranate', 'pomelo', 'pool table', 'porcupine', 'potato',
                        'potato chips', 'potted plant', 'power drill', 'prairie chicken', 'prayer rug', 'pretzel', 'printer', 'proboscis', 'projector', 'psp', 'pspgo', 'ptarmigan', 'pteropus',
                        'pubg airdrop', 'pubg lvl3backpack', 'pubg lvl3helmet', 'pufferfish', 'puma logo', 'pumpkin', 'pumpkin pie', 'punching bag', 'pyramid', 'pyramid cube', 'pyraminx',
                        'quad drone', 'quail', 'quail egg', 'quill pen', 'rabbit', 'raccoon', 'radiator', 'radio', 'radio telescope', 'raft', 'rain barrel', 'rally car', 'raven', 'razor',
                        'recreational vehicle', 'redheart', 'redshank', 'red bayberry', 'red breasted merganser', 'red fox', 'red wolf', 'reel', 'reflex camera', 'refrigerator',
                        'relay stick', 'remote control', 'revolver', 'rhinoceros', 'rice cooker', 'ringlet butterfly', 'rocket', 'rocking chair', 'rock beauty', 'rock snake',
                        'roller coaster', 'roller skate', 'rose', 'rosehip', 'rubber eraser', 'rubick cube', 'ruddy turnstone', 'ruffed grouse', 'rugby ball', 'ruler',
                        'running shoe', 'saltshaker', 'saluki', 'samarra mosque', 'sandal', 'sandbar', 'sandwich', 'sandwich cookies', 'santa sledge', 'sarong', 'saxophone',
                        'scabbard', 'scarerow', 'school bus', 'schooner', 'scissors', 'scorpion', 'screw', 'screwdriver', 'scroll brush', 'seagull', 'seal', 'sealion',
                        'seatbelt', 'sea cucumber', 'sea urchin', 'sewing machine', 'shakuhachi', 'shift gear', 'shih-tzu', 'shopping cart', 'shotgun', 'shovel',
                        'shower cap', 'shower curtain', 'shumai', 'shuriken', 'siamang', 'siamese cat', 'sidewinder', 'single log', 'skateboard', 'ski mask', 'skua',
                        'skull', 'skunk', 'sled', 'sleeping bag', 'sloth bear', 'smoothing iron', 'snail', 'snake', 'sniper rifle', 'snowball', 'snowman', 'snowmobile',
                        'snowplow', 'snow leopard', 'soap', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'soymilk machine',
                        'space heater', 'space shuttle', 'spade', 'spark plug', 'sparrow', 'spatula', 'speaker', 'speedboat', 'spider', 'spiderman', 'spider monkey',
                        'spinach', 'sponge', 'spoon', 'spoonbill', 'sports car', 'spotted salamander', 'spring scroll', 'squirrel', 'squirrel monkey', 'staffordshire',
                        'stapler', 'starfish', 'statue liberty', 'steak', 'stealth aircraft', 'steam locomotive', 'steering wheel', 'stingray', 'stinkhorn', 'stole',
                        'stonechat', 'stone lion', 'stool', 'stopwatch', 'stop sign', 'stork', 'stove', 'strainer', 'strawberry', 'streetcar', 'stretcher', 'strongbox',
                        'studio couch', 'stupa', 'sturgeon', 'submarine', 'suitcase', 'sulphur butterfly', 'sulphur crested', 'sundial', 'sunglasses', 'sungnyemun',
                        'sunscreen', 'surfboard', 'sushi', 'swab', 'swan', 'sweatshirt', 'swimming glasses', 'swimming trunk', 'swim ring', 'sydney opera house', 'syringe',
                        'table lamp', 'taj mahal', 'tank', 'tape player', 'taro', 'taxi', 'teapot', 'tebby cat', 'teddy', 'telescope', 'television', 'tennis racket',
                        'terrapin turtle', 'thatch', 'thimble', "thor's hammer", 'three-toed sloth', 'throne', 'thrush', 'tiger', 'tiger cat',
                        'tiger shark', 'tile roof', 'tiltrotor', 'timber wolf', 'titi monkey', 'toaster', 'tobacco pipe', 'tofu', 'toilet brush', 'toilet plunger',
                        'toilet seat', 'toilet tissue', 'tokyo tower', 'tomato', 'tomb', 'toothbrush', 'toothpaste', 'torii', 'totem pole', 'toucan', 'tower pisa',
                        'tow truck', 'tractor', 'traffic light', 'trailer truck', 'transport helicopter', 'tray', 'tredmill', 'trench coat', 'tresher', 'triceratops',
                        'trilobite', 'trimaran', 'triumphal arch', 'trolleybus', 'truss bridge', 'tulip', 'tunnel', 'turnstile', 'turtle', 'twin tower', 'typewriter',
                        'umbrella', 'upright piano', 'usb', 'vacuum', 'vacuum cup', 'vase', 'vending machine', 'vestment', 'victor icon', 'villa savoye', 'vine snake',
                        'vinyl', 'violin', 'volleyball', 'vulture', 'wafer', 'waffle', 'waffle iron', 'wagtail', 'wallaby', 'wallet', 'wall clock', 'walnut',
                        'wandering albatross', 'wardrobe', 'warehouse tray', 'warplane', 'warthog', 'washer', 'wash basin', 'wasp', 'watermelon', 'water bike',
                        'water buffalo', 'water heater', 'water ouzel', 'water polo', 'water snake', 'water tower', 'weasel', 'whale', 'wheelchair', 'whippet', 'whiptail',
                        'whistle', 'white shark', 'white stork', 'white wolf', 'wig', 'wild boar', 'windmill', 'window screen', 'window shade', 'windsor tie', 'wine bottle',
                        'witch hat', 'wok', 'wolf', 'wombat', 'wooden boat', 'wooden spoon', 'woodpecker', 'wreck', 'wrench', 'yawl', 'yoga pad', 'yonex icon',
                        'yorkshire terrier', 'yurt', 'zebra', 'zucchini']


new_class_names_fss = ['abacus', "abe's flyingfish", 'ab wheel', 'accordion', 'acorn', 'ac ground', 'ac wall', 'adhensive tape',
                        'adidas logo1', 'adidas logo2', 'afghan hound', 'african crocodile', 'african elephant', 'african grey',
                        'agama', 'aircraft carrier', 'airedale', 'airliner', 'airship', 'air strip', 'albatross', 'almond', 'ambulance',
                        'american alligator', 'american chamelon', 'american staffordshire', 'andean condor', 'anemone fish', 'angora',
                        'anise', 'apple', 'apple icon', 'apron', 'arabian camel', 'arch bridge', 'arctic fox', 'armadillo', 'armour',
                        'arrow', 'artichoke', 'ashtray', 'assult rifle', 'astronaut', 'aubergine', 'australian terrier', 'avocado', 'baboon',
                        'baby', 'backpack', 'badger', 'bagel', 'balance beam', 'balance weight', 'bald eagle', 'balloon', 'ballpoint', 'bamboo dragonfly',
                        'bamboo slip', 'banana', 'banana boat', 'band-aid', 'banded gecko', 'banjo', 'barbell', 'barber shaver',
                        'barometer', 'baseball', 'baseball bat', 'baseball player', 'basketball', 'basset', 'bassoon', 'bat',
                        'bathtub', 'bath ball', 'battery', 'beacon', 'beagle', 'beaker', 'beam bridge', 'bear', 'beaver', 'bedlington terrier',
                        'bee', 'beer bottle', 'beer glass', 'beet root', 'bee eater', 'bee house', 'bell', 'bell pepper', 'besom', 'bighorn sheep',
                        'big ben', 'birdhouse', 'bison', 'bittern', 'black bear', 'black grouse', 'black stork', 'black swan', 'blenheim spaniel',
                        'bloodhound', 'blossom card', 'bluetick', 'boa constrictor', 'bolete', 'bolotie', 'bomb', 'border terrier', 'boston bull',
                        'bottle cap', 'bouzouki', 'bowtie', 'boxing gloves', 'box turtle', 'bra', 'bracelet', 'bradypod', 'brain coral', 'brambling',
                        'brasscica', 'breast pump', 'briard', 'brick', 'brick card', 'brick tea', 'briefcase', 'broccoli', 'broom', 'brown bear',
                        'brush pen', 'bucket', 'buckingham palace', 'buckler', 'bulb', 'bulbul bird', 'bullet train', 'burj al', 'bus', 'bushtit',
                        'bustard', 'butterfly', 'cabbage', 'cabbage butterfly', 'cablestayed bridge', 'cableways', 'cactus', 'cactus ball', 'cairn',
                        'calculator', 'camel', 'camomile', 'candle', 'cannon', 'canoe', 'cantilever bridge', 'canton tower', 'can opener', 'captain america shield',
                        'capuchin', 'carambola', 'carbonara', 'cardoon', 'carousel', 'carp', 'carriage', 'carrot', 'carton', 'car mirror', 'car wheel', 'cassette',
                        'cathedrale paris', 'cauliflower', 'cd', 'ceiling fan', 'celery', 'cello', 'chainsaw', 'chalk', 'chalk brush', 'chandelier', 'charge battery',
                        'cheese', 'cheese burger', 'cheetah', 'cherry', 'chess bishop', 'chess king', 'chess knight', 'chess queen', 'chest', 'chickadee bird',
                        'chicken', 'chicken leg', 'chicken wings', 'chicory', 'chiffon cake', 'chihuahua', 'children slide', 'chimpanzee', 'chinese date',
                        'chinese knot', 'chopsticks', 'christmas stocking', 'church', 'cicada', 'cigar', 'cigarette', 'clam', 'clearwing flyingfish', 'cleaver',
                        'cloud', 'cn tower', 'cocacola', 'cocktail shaker', 'coconut', 'coffeepot', 'coffee mug', 'coffin', 'coho', 'coin', 'collar', 'colubus',
                        'comb', 'combination lock', 'common newt', 'computer mouse', 'conch', 'condor', 'consomme', 'conversion plug', 'convertible', 'conveyor',
                        'corn', 'cornet', 'cornmeal', 'cosmetic brush', 'cottontail', 'coucal', 'cougar', 'cowboy hat', 'coyote', 'cpu', 'crab', 'cradle', 'crane',
                        'crash helmet', 'crayon', 'cream', 'crepe', 'cricket', 'cricketball', 'cristo redentor', 'crocodile', 'croissant', 'croquet ball',
                        'crt screen', 'cuckoo', 'cucumber', 'cumquat', 'cup', 'curlew', 'cushion', 'daisy', 'dandie dinmont', 'dart', 'dart target', 'delta wing',
                        'dhole', 'diamond', 'diaper', 'digital clock', 'digital watch', 'dingo', 'dinosaur', 'dishwasher', 'diver', 'donkey', 'doormat', 'doublebus',
                        'dough', 'doughnut', 'dowitcher', 'downy pitch', 'dragonfly', 'drake', 'drilling platform', 'drum', 'drumstick', 'dugong', 'dumbbell', 'dutch oven',
                        'dwarf beans', 'eagle', 'earphone1', 'earphone2', 'earplug', 'echidna', 'eel', 'eft newt', 'egg', 'eggnog', 'egg tart', 'egret', 'egyptian cat',
                        'electric fan', 'electronic stove', 'electronic toothbrush', 'eletrical switch', 'english foxhound', 'english setter', 'envelope', 'equestrian helmet',
                        'esport chair', 'espresso', 'espresso maker', 'excavator', 'f1 racing', 'face powder', 'fan', 'feather clothes', 'feeder', 'fennel bulb', 'ferrari911',
                        'ferret', 'fig', 'file cabinet', 'fire balloon', 'fire engine', 'fire hydrant', 'fire screen', 'fish', 'fish eagle', 'flamingo', 'flat-coated retriever',
                        'flatworm', 'flowerpot', 'flute', 'fly', 'flying disc', 'flying frog', 'flying geckos', 'flying snakes', 'flying squirrel', 'folding chair', 'fork', 'forklift',
                        'fountain', 'fox', 'fox squirrel', 'french ball', 'french fries', 'frog', 'frying pan', 'fur coat', 'ganeva chair', 'garbage can', 'garbage truck', 'garfish',
                        'garlic', 'gas pump', 'gas tank', 'gazelle', 'gecko', 'german pointer', 'giant panda', 'giant schnauzer', 'gibbon', 'ginger', 'glider flyingfish', 'gliding lizard',
                        'globe', 'goblet', 'golden plover', 'golden retriever', 'goldfinch', 'goldfish', 'golfcart', 'golf ball', 'goose', 'gorilla', 'gourd', 'grasshopper', 'great wall',
                        'green mamba', 'grey fox', 'grey whale', 'groenendael', 'guacamole', 'guinea pig', 'guitar', 'gym ball', 'gypsy moth', 'gyromitra', 'haddock', 'hair drier',
                        'hair razor', 'hami melon', 'hammer', 'hammerhead shark', 'hamster', 'handcuff', 'handkerchief', 'handshower', 'hang glider', 'hard disk', 'hare', 'harmonica',
                        'harp', 'hartebeest', 'harvester', 'har gow', 'hatchet', 'hawk', 'hawthorn', 'head cabbage', 'helicopter', 'hen of the woods', 'hippo', 'hock', 'hook', 'hornbill',
                        'hornet', 'hotdog', 'hotel slipper', 'housefinch', 'hover board', 'howler monkey', 'hummingbird', 'hyena', 'ibex', 'iceberg', 'icecream', 'ice lolly', 'igloo',
                        'iguana', 'impala', 'indian cobra', 'indian elephant', 'indri', 'ipad', 'iphone', 'ipod', 'ironing board', 'iron man', 'jacamar', 'jackfruit', 'jacko lantern',
                        'jay bird', 'jellyfish', 'jet aircraft', 'jinrikisha', 'jordan logo', 'joystick', 'kangaroo', 'kappa logo', 'kart', 'kazoo', 'key', 'keyboard', 'killer whale',
                        'kinguin', 'kitchen knife', 'kite', 'kit fox', 'knife', 'koala', 'kobe logo', 'kremlin', 'kunai', 'kwanyin', 'lacewing', 'ladder', 'ladle', 'ladybug', 'ladyfinger',
                        'lady slipper', 'lampshade', 'langur', 'laptop', 'lapwing', 'lark', 'lawn mower', 'leafhopper', 'leaf egg', 'leaf fan', 'leatherback turtle', 'leather shoes',
                        'leeks', 'leggings', 'lemon', 'lemur catta', 'leopard', 'lesser panda', 'letter opener', 'lettuce', 'lhasa apso', 'lifeboat', 'light tube', 'lion', 'lionfish',
                        'lipstick', 'litchi', 'little blue heron', 'llama', 'loafer', 'lobster', 'loggerhead turtle', 'loguat', 'lorikeet', 'lotus', 'louvre pyramid', 'lycaenid butterfly',
                        'lynx', 'macaque', 'macaw', 'magpie bird', 'mailbox', 'manatee', 'mango', 'manx', 'maotai bottle', 'maraca', 'marimba', 'mario', 'marmot', 'marshmallow',
                        'mashed potato', 'matchstick', 'may bug', 'mcdonald sign', 'mcdonald uncle', 'measuring cup', 'meatloaf', 'medical kit', 'meerkat', 'melon seed', 'memory stick',
                        'microphone', 'microscope', 'microsd', 'microwave', 'military vest', 'minicooper', 'miniskirt', 'mink', 'missile', 'mite predator', 'mitten', 'modem', 'moist proof pad',
                        'monarch butterfly', 'mongoose', 'monitor', 'monkey', 'monocycle', 'mooli', 'moon', 'mooncake', 'mortar', 'motarboard', 'motorbike', 'motor scooter', 'mountain tent',
                        'mount fuji', 'mouse', 'mouthpiece', 'mud turtle', 'mule', 'muscle car', 'mushroom', 'nagoya castle', 'nail scissor', 'narcissus', 'necklace', 'neck brace', 'nematode',
                        'net surface shoes', 'night snake', 'nike logo', 'nintendo 3ds', 'nintendo gba', 'nintendo sp', 'nintendo switch', 'nintendo wiiu', 'obelisk', 'ocarina', 'ocicat',
                        'oiltank car', 'oil filter', 'okra', 'olive', 'one-armed bandit', 'onion', 'orang', 'orange', 'oriole', 'oscilloscope', 'osprey', 'ostrich', 'otter', 'owl', 'ox',
                        'oyster', 'paddle', 'paint brush', 'panda', 'panpipe', 'panther', 'papaya', 'paper crane', 'paper plane', 'paper towel', 'parachute', 'parallel bars', 'parking meter',
                        'park bench', 'parthenon', 'partridge', 'patas', 'pay phone', 'peacock', 'peanut', 'pear', 'pen', 'pencil box', 'pencil sharpener1', 'pencil sharpener2', 'pepitas',
                        'peregine falcon', 'perfume', 'persian cat', 'persimmon', 'petri dish', 'pheasant', 'phonograph', 'photocopier', 'piano keyboard', 'pickelhaube', 'pickup', 'pidan',
                        'pig', 'pillow', 'pill bottle', 'pineapple', 'pinecone', 'pingpong ball', 'pingpong racket', 'pinwheel', 'pistachio', 'pizza', 'plaice', 'plastic bag', 'plate',
                        'platypus', 'poached egg', 'poker', 'pokermon ball', 'polar bear', 'polecat', 'police car', 'polo shirt', 'pomegranate', 'pomelo', 'pool table', 'porcupine', 'potato',
                        'potato chips', 'potted plant', 'power drill', 'prairie chicken', 'prayer rug', 'pretzel', 'printer', 'proboscis', 'projector', 'psp', 'pspgo', 'ptarmigan', 'pteropus',
                        'pubg airdrop', 'pubg lvl3backpack', 'pubg lvl3helmet', 'pufferfish', 'puma logo', 'pumpkin', 'pumpkin pie', 'punching bag', 'pyramid', 'pyramid cube', 'pyraminx',
                        'quad drone', 'quail', 'quail egg', 'quill pen', 'rabbit', 'raccoon', 'radiator', 'radio', 'radio telescope', 'raft', 'rain barrel', 'rally car', 'raven', 'razor',
                        'recreational vehicle', 'redheart', 'redshank', 'red bayberry', 'red breasted merganser', 'red fox', 'red wolf', 'reel', 'reflex camera', 'refrigerator',
                        'relay stick', 'remote control', 'revolver', 'rhinoceros', 'rice cooker', 'ringlet butterfly', 'rocket', 'rocking chair', 'rock beauty', 'rock snake',
                        'roller coaster', 'roller skate', 'rose', 'rosehip', 'rubber eraser', 'rubick cube', 'ruddy turnstone', 'ruffed grouse', 'rugby ball', 'ruler',
                        'running shoe', 'saltshaker', 'saluki', 'samarra mosque', 'sandal', 'sandbar', 'sandwich', 'sandwich cookies', 'santa sledge', 'sarong', 'saxophone',
                        'scabbard', 'scarerow', 'school bus', 'schooner', 'scissors', 'scorpion', 'screw', 'screwdriver', 'scroll brush', 'seagull', 'seal', 'sealion',
                        'seatbelt', 'sea cucumber', 'sea urchin', 'sewing machine', 'shakuhachi', 'shift gear', 'shih-tzu', 'shopping cart', 'shotgun', 'shovel',
                        'shower cap', 'shower curtain', 'shumai', 'shuriken', 'siamang', 'siamese cat', 'sidewinder', 'single log', 'skateboard', 'ski mask', 'skua',
                        'skull', 'skunk', 'sled', 'sleeping bag', 'sloth bear', 'smoothing iron', 'snail', 'snake', 'sniper rifle', 'snowball', 'snowman', 'snowmobile',
                        'snowplow', 'snow leopard', 'soap', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'soymilk machine',
                        'space heater', 'space shuttle', 'spade', 'spark plug', 'sparrow', 'spatula', 'speaker', 'speedboat', 'spider', 'spiderman', 'spider monkey',
                        'spinach', 'sponge', 'spoon', 'spoonbill', 'sports car', 'spotted salamander', 'spring scroll', 'squirrel', 'squirrel monkey', 'staffordshire',
                        'stapler', 'starfish', 'statue liberty', 'steak', 'stealth aircraft', 'steam locomotive', 'steering wheel', 'stingray', 'stinkhorn', 'stole',
                        'stonechat', 'stone lion', 'stool', 'stopwatch', 'stop sign', 'stork', 'stove', 'strainer', 'strawberry', 'streetcar', 'stretcher', 'strongbox',
                        'studio couch', 'stupa', 'sturgeon', 'submarine', 'suitcase', 'sulphur butterfly', 'sulphur crested', 'sundial', 'sunglasses', 'sungnyemun',
                        'sunscreen', 'surfboard', 'sushi', 'swab', 'swan', 'sweatshirt', 'swimming glasses', 'swimming trunk', 'swim ring', 'sydney opera house', 'syringe',
                        'table lamp', 'taj mahal', 'tank', 'tape player', 'taro', 'taxi', 'teapot', 'tebby cat', 'teddy', 'telescope', 'television', 'tennis racket',
                        'terrapin turtle', 'thatch', 'thimble', "thor's hammer", 'three-toed sloth', 'throne', 'thrush', 'tiger', 'tiger cat',
                        'tiger shark', 'tile roof', 'tiltrotor', 'timber wolf', 'titi monkey', 'toaster', 'tobacco pipe', 'tofu', 'toilet brush', 'toilet plunger',
                        'toilet seat', 'toilet tissue', 'tokyo tower', 'tomato', 'tomb', 'toothbrush', 'toothpaste', 'torii', 'totem pole', 'toucan', 'tower pisa',
                        'tow truck', 'tractor', 'traffic light', 'trailer truck', 'transport helicopter', 'tray', 'tredmill', 'trench coat', 'tresher', 'triceratops',
                        'trilobite', 'trimaran', 'triumphal arch', 'trolleybus', 'truss bridge', 'tulip', 'tunnel', 'turnstile', 'turtle', 'twin tower', 'typewriter',
                        'umbrella', 'upright piano', 'usb', 'vacuum', 'vacuum cup', 'vase', 'vending machine', 'vestment', 'victor icon', 'villa savoye', 'vine snake',
                        'vinyl', 'violin', 'volleyball', 'vulture', 'wafer', 'waffle', 'waffle iron', 'wagtail', 'wallaby', 'wallet', 'wall clock', 'walnut',
                        'wandering albatross', 'wardrobe', 'warehouse tray', 'warplane', 'warthog', 'washer', 'wash basin', 'wasp', 'watermelon', 'water bike',
                        'water buffalo', 'water heater', 'water ouzel', 'water polo', 'water snake', 'water tower', 'weasel', 'whale', 'wheelchair', 'whippet', 'whiptail',
                        'whistle', 'white shark', 'white stork', 'white wolf', 'wig', 'wild boar', 'windmill', 'window screen', 'window shade', 'windsor tie', 'wine bottle',
                        'witch hat', 'wok', 'wolf', 'wombat', 'wooden boat', 'wooden spoon', 'woodpecker', 'wreck', 'wrench', 'yawl', 'yoga pad', 'yonex icon',
                        'yorkshire terrier', 'yurt', 'zebra', 'zucchini']

BACKGROUND_CATEGORY_fss = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge']
