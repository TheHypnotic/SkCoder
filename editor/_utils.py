import json


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item
    # print(f"🚀 Loading data from: {filename}")

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
            # print('first if ')
        else:
            source_str = "{}: {}".format(args.task, example.source)
            # print('else')
    else:
        # print('first else')
        source_str = example.source
        # prompt = "Write a Python class that represents a Legendary minion card with the following properties: - Name: Vol'jin - Type: Minion - Class: Priest - Attack: 6 - Defense: 5 - Cost: -1 (This might indicate the card is free or has special cost mechanics) - Special Ability: Battlecry - Swap Health with another minion The class should inherit from `MinionCard` and use the following attributes and methods: - `__init__(self)`: Initialize the card with the name, attack, defense, cost, and special ability. - `create_minion(self, player)`: Create a minion using the specified attributes. - The class should include any necessary imports and handle edge cases, such as ensuring valid player input for abilities and health swaps. The class should also include a simple representation of the card for debugging or logging purposes. Please provide this implementation in a clean, Pythonic format with proper comments."
        # source_str += prompt 
        # print(example.oracle_sketch)
        if hasattr(example, "oracle_sketch"):  # Ensure sketch is included if it exists
            # print('second if')
            max_sketch_items = 1
            # sketch_str = " ".join(example.oracle_sketch[:max_sketch_items])
            # sketch_str = " ".join(example.oracle_sketch[2:3])
            sketch_str = " ".join(example.oracle_sketch)
            source_str += " [SEP] " + sketch_str
            # print(sketch_str)
            # raise ValueError(sketch_str)



    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task='', 
                 sketch=None, 
                 oracle_sketch=None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task
        self.sketch = sketch
        self.oracle_sketch = oracle_sketch


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_CG_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            js = json.loads(line.strip())
            if 'input_tokens' in js and 'output_tokens' in js:
                source = ' '.join(js['input_tokens'])
                target = ' '.join(js['output_tokens'])
            elif 'input' in js and 'output' in js:
                source = js['input']
                target = js['output']
            if "oracle-sketch" in js:
                oracle_sketch = js["oracle-sketch"]
            else:
                oracle_sketch = None

            examples.append(
                Example(
                    idx=idx,
                    source=source,
                    target=target,
                    sketch=js.get("sketch", None),
                    oracle_sketch=oracle_sketch)
            )
            if idx+1 == data_num:
                break
    return examples