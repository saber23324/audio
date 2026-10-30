"""Label-name-only source category proxy and GT-free FullVision agreement."""
from .heatmaps import rng_for


# Frozen before any image/model output. Only official class strings are used;
# ambiguous actions/environment sounds remain not_applicable. These labels
# are semantic proxies, not independently annotated frame-level QA answers.
GROUPS = {
 'cattle': ['cow lowing','bull bellowing','cattle, bovinae cowbell','cattle mooing'],
 'donkey': ['donkey, ass braying'],
 'bird': ['cuckoo bird calling','bird chirping, tweeting','pheasant crowing','canary calling',
          'bird wings flapping','warbler chirping','owl hooting','magpie calling','wood thrush calling',
          'barn swallow calling','francolin calling','crow cawing','woodpecker pecking tree'],
 'chicken': ['chicken clucking','chicken crowing'], 'turkey': ['turkey gobbling'],
 'penguin': ['penguins braying'], 'snake': ['snake hissing','snake rattling'],
 'whale': ['whale calling'], 'dog': ['dog barking','dog growling','dog bow-wow','dog howling'],
 'cat': ['cat growling','cat meowing','cat hissing'], 'otter': ['otter growling'],
 'goat': ['goat bleating'], 'coyote': ['coyote howling'], 'fox': ['fox barking'],
 'lion': ['lions roaring','lions growling'], 'crocodile': ['alligators, crocodiles hissing'],
 'chipmunk': ['chipmunk chirping'], 'gibbon': ['gibbon howling'],
 'airplane': ['airplane','airplane flyby'], 'helicopter': ['helicopter'],
 'train': ['train wheels squealing','railroad car, train wagon','train whistling','subway, metro, underground'],
 'car': ['car passing by','race car, auto racing'], 'motorcycle': ['driving motorcycle'],
 'fire truck': ['fire truck siren'], 'motorboat': ['motorboat, speedboat acceleration'],
 'snowmobile': ['driving snowmobile'], 'bus': ['driving buses'], 'tractor': ['tractor digging'],
 'typewriter': ['typing on typewriter'], 'sewing machine': ['using sewing machines'],
 'vacuum cleaner': ['vacuum cleaner cleaning floors'], 'alarm clock': ['alarm clock ringing'],
 'wind chime': ['wind chime'], 'lawn mower': ['lawn mowing'],
}
CLASS_MAP = {label: category for category, labels in GROUPS.items() for label in labels}


def category_task(label, source_id, config):
    category = CLASS_MAP.get(label)
    if category is None:
        return {'applicable': False, 'reason': 'abstract_environment_action_or_not_whitelisted_class'}
    rng = rng_for(source_id, 'category-options', config['seed'])
    pool = sorted(set(GROUPS) - {category})
    options = [category, *rng.choice(pool, 3, replace=False).tolist()]
    rng.shuffle(options)
    return {'applicable': True, 'task': 'VGGSS-derived source-category proxy',
            'question': config['category_question'], 'options': [f'{letter}. {text}' for letter, text in zip('ABCD', options)],
            'answer': 'ABCD'[options.index(category)], 'category': category,
            'answer_provenance': 'fixed mapping of official sound-event class name; not official VGGSS QA'}


def teacher_task(config):
    return {'question': config['teacher_question'], 'options': config['teacher_options'],
            'task': 'FullVision prediction agreement; teacher is not ground truth'}
