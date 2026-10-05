"""The direct-membership prompt used in the A6 objective experiments (X8, X9)."""
import json


def member_prompt(row):
    return ('Does this context-free grammar generate exactly the given input token sequence? '
        'Start from the grammar start symbol and apply its productions. '
        'Return {"accept": true} if some derivation generates the input, otherwise {"accept": false}. '
        'Return only that raw JSON object, without Markdown or explanation.\n'+
        json.dumps({'grammar':row['grammar'],'input':row['input']},sort_keys=True))
