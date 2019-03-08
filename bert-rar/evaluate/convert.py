import json

with open('your predictions file', 'r') as reader:
    source = json.load(reader)

f = open('output file', 'w')
dic = {'answers': ['tmp'], 'query_id': 0}
for key in source.keys():
    dic['answers'][0] = source[key]
    dic['query_id'] = int(key)
    json.dump(dic, f)
    f.write('\n')
f.close()
