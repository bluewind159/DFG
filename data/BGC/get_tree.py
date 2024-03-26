f=open('./hierarchy.txt','r',encoding='utf-8')
data=f.readlines()
parents=[]
childs=[]
for line in data:
    l=line.strip().split('\t')
    if len(l)==2:
        parent,child=l
        parents.append(parent)
        childs.append(child)
    else:
        parent=l[0]
        parents.append(parent)
parents=list(set(parents))
childs=list(set(childs))
taxonomy={}
taxonomy['Root']=set()
for pa in parents:
    if pa not in childs:
       taxonomy['Root'].add(pa)
for line in data:
    l=line.strip().split('\t')
    if len(l)==2:
        parent,child=l
        if parent not in taxonomy:
            taxonomy[parent] = set()
        taxonomy[parent].add(child)
label_hierarchy=taxonomy
print(label_hierarchy.keys())
f = open('bgc.taxonomy_new', 'w')
keys=[['Root'],[],[],[],[],[],[]]
for kk in range(len(keys)-1):
    for key233 in keys[kk]:
        for i233 in label_hierarchy.keys():
            if i233 in label_hierarchy[key233]:
                keys[kk+1].append(i233)
final=[]
for key in keys:
    final = final+key
print(len(final))
print(len(label_hierarchy.keys()))
for i in final:
    line = [i]
    line.extend(sorted(label_hierarchy[i],key=lambda x:x[0]))
    line = '\t'.join(line) + '\n'
    f.write(line)
f.close()