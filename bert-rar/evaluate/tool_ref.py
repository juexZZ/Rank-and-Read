import json

QUERY_ID_JSON_ID = 'query_id'
ANSWERS_JSON_ID = 'answers'

def load_file(p_path_to_data):
    all_answers = []
    query_ids = []
    with open(p_path_to_data, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            try:
                json_object = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)

            assert \
                QUERY_ID_JSON_ID in json_object, \
                '\"%s\" json does not have \"%s\" field' % \
                    (line, QUERY_ID_JSON_ID)
            query_id = json_object[QUERY_ID_JSON_ID]

            assert \
                ANSWERS_JSON_ID in json_object, \
                '\"%s\" json does not have \"%s\" field' % \
                    (line, ANSWERS_JSON_ID)
            answers = json_object[ANSWERS_JSON_ID]
            if 'No Answer Present.' in answers:
                no_answer_query_ids.add(query_id)
                answers = ['']
            all_answers.extend(answers)
            query_ids.extend([query_id]*len(answers))

    query_id_to_answers_map = {}
    for i, normalized_answer in enumerate(all_answers):
        query_id = query_ids[i]
        if query_id not in query_id_to_answers_map:
            query_id_to_answers_map[query_id] = []
        query_id_to_answers_map[query_id].append(normalized_answer)
    return query_id_to_answers_map

maps = load_file('output file of convert.py')

with open('marcoms dev_v2.1.json file') as f:
    source = json.load(f)

query_ids = source['query_id']
queries = source['query']
passages = source['passages']
answers = source.get('answers', {})

fout = open('output file', 'w')
for key in maps.keys():
    key = str(key)
    if key in query_ids.keys():
        json.dump({'answers':answers.get(key), 'query_id':int(key)}, fout)
        fout.write('\n')
    else:
        print(key)
fout.close()
